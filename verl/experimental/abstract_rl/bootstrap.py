# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Two-phase format bootstrap.

Some policies never emit ``<abstract>`` at all. Measured on
Qwen2.5-Math-1.5B-Instruct: 0 of 416 sampled rollouts across six prompt
placements -- it stops at ``<|im_end|>`` immediately after ``\\boxed{}``. With no
valid abstraction every ``U_i`` is 0, every group is flat, and RL has nothing to
amplify: ``grad_norm`` is exactly 0 for the whole run.

This bootstraps the format out of the policy itself rather than importing it from
a second model. For the first N steps, rollouts that produced no abstraction get a
*second, short* generation conditioned on their own question and solution, and the
result is spliced into the response as a proper tagged block. GRPO then raises the
likelihood of the whole grafted sequence under the training prompt whenever it
scores well -- reward-weighted self-distillation of the format. Once the policy
emits abstractions on its own, grafting switches off and training reverts to
ordinary single-stage GRPO.

Two things about this are worth being explicit about.

*It is a second generation stage*, which the method otherwise forbids. It is off by
default, gated to the bootstrap window, logged every step
(``abstract_rl/bootstrap_active``), and self-disables.

*During bootstrap the update is not a strict policy gradient.* A grafted sequence
was not sampled from ``pi_theta(.|X_train)``, so the ratio is 1 by construction and
the step behaves as reward-weighted self-distillation rather than exact GRPO. That
is the intent, and it is why it has to switch off before the numbers mean anything.

Only a *fraction* of eligible rollouts is grafted (``bootstrap_graft_fraction``),
and this is the part that makes it work at all. If every rollout in a group carried
an abstraction, the tags would appear in positive- and negative-advantage sequences
alike and the gradient on *emitting* them would largely cancel -- the policy would
learn abstraction quality but never abstraction presence. Leaving some rollouts
untagged gives them ``v_i = 0``, hence ``U_i = 0``, hence a strictly lower reward
than their tagged siblings. That contrast is the format signal.
"""

from __future__ import annotations

import re
from collections import deque

import numpy as np
import torch

from verl import DataProto

from .config import AbstractRLConfig
from .parsing import AbstractSpan

PHASE2_TEMPLATE = (
    "{question}\n\n{solution}\n\n"
    "In one short sentence, name the type of problem. In one more, give the general "
    "method, with no numbers. Then stop."
)


# Steps averaged on each side of the accuracy guard. 3 is enough to kill the
# ~0.05 per-step swing of a 64-prompt batch without delaying a real abort much.
# 5, not 3: train accuracy on a 64-prompt batch has std ~0.052 step to step, so a
# 3-step mean has std ~0.030 and the default 0.05 tolerance is a <2-sigma trip --
# it fired on pure noise at step 7 of 30 in abstractrl-warm30-20260806. At 5 the
# window mean has std ~0.023 and the guard cannot arm before step 10.
_ACC_WINDOW = 5


class BootstrapState:
    """Tracks native compliance so grafting can switch itself off."""

    def __init__(self, cfg: AbstractRLConfig):
        self.cfg = cfg
        self.consecutive_above = 0
        self.disabled = False
        self.baseline_accuracy: float | None = None
        self.baseline_abstract_len: float | None = None
        self._acc_window: deque[float] = deque(maxlen=2 * _ACC_WINDOW)
        self.sft_disabled = False
        self.sft_abort_reason = ""

    def observe_abstract_length(self, mean_tokens: float) -> bool:
        """Abort the warm-up if the abstractions are collapsing toward nothing.

        Length is the LEADING indicator of the self-distillation runaway; accuracy
        lags it. Measured at lr 1e-5, mean abstract tokens went 79 -> 49 -> 3.7 while
        accuracy had only moved 0.56 -> 0.52, so an accuracy-only guard fires late.
        The floor is a fraction of the length the *pre-warm-up* policy produced, not
        an absolute count, because a genuinely more concise abstract is the goal --
        this only catches the run to zero.
        """
        frac = self.cfg.bootstrap_sft_min_length_fraction
        if self.sft_disabled or self.cfg.bootstrap_sft_steps <= 0 or frac <= 0 or mean_tokens <= 0:
            return self.sft_disabled
        if self.baseline_abstract_len is None:
            self.baseline_abstract_len = mean_tokens
        elif mean_tokens < self.baseline_abstract_len * frac:
            self.sft_disabled = True
            self.sft_abort_reason = "length"
            print(
                f"NOTICE: abstract_rl SFT warm-up aborted -- mean abstraction length {mean_tokens:.1f} "
                f"is below {frac:.2f} x the pre-warm-up baseline {self.baseline_abstract_len:.1f}. "
                "The self-distillation loop is collapsing; falling back to GRPO."
            )
        return self.sft_disabled

    def observe_accuracy(self, accuracy: float) -> bool:
        """Abort the SFT warm-up if it is eating the solver.

        The warm-up loss is masked to the abstraction span, but the gradient still
        lands on shared weights, so it can degrade the solver.

        Both sides are means over ``_ACC_WINDOW`` steps, not single observations.
        Train accuracy on a 64-prompt batch swings ~0.05 step to step on its own,
        and thresholding raw values fired on noise once already: a 0.570 -> 0.443
        "drop" was back to 0.551 the very next step, and held-out math500 pass@1
        was 0.644 against 0.646 for the reference run -- no damage at all, but the
        warm-up had already been killed at step 5 of 30.

        Once tripped it stays tripped.
        """
        tol = self.cfg.bootstrap_sft_accuracy_drop
        if self.sft_disabled or self.cfg.bootstrap_sft_steps <= 0 or tol <= 0:
            return self.sft_disabled

        self._acc_window.append(accuracy)
        if len(self._acc_window) < 2 * _ACC_WINDOW:
            return False  # still establishing the baseline
        window = list(self._acc_window)
        baseline = sum(window[:_ACC_WINDOW]) / _ACC_WINDOW
        recent = sum(window[-_ACC_WINDOW:]) / _ACC_WINDOW
        self.baseline_accuracy = baseline
        if recent < baseline - tol:
            self.sft_disabled = True
            self.sft_abort_reason = "accuracy"
            print(
                f"NOTICE: abstract_rl SFT warm-up aborted -- train accuracy averaged {recent:.4f} over the "
                f"last {_ACC_WINDOW} steps, more than {tol} below the {baseline:.4f} of the first "
                f"{_ACC_WINDOW}. Falling back to GRPO; grafting continues."
            )
        return self.sft_disabled

    def active(self, global_step: int) -> bool:
        if self.disabled or self.cfg.bootstrap_steps <= 0:
            return False
        return global_step <= self.cfg.bootstrap_steps

    def observe(self, native_rate: float) -> None:
        """Feed the pre-graft native emission rate; latch off once it holds up."""
        if native_rate >= self.cfg.bootstrap_native_threshold:
            self.consecutive_above += 1
            if self.consecutive_above >= self.cfg.bootstrap_patience:
                self.disabled = True
        else:
            self.consecutive_above = 0


def thin_rows(eligible: list[int], fraction: float, seed: int) -> list[int]:
    """Randomly keep ``fraction`` of ``eligible``, deterministic given ``seed``."""
    if not eligible:
        return []
    rng = np.random.default_rng(seed)
    keep = max(0, min(int(round(len(eligible) * fraction)), len(eligible)))
    return sorted(rng.choice(eligible, size=keep, replace=False).tolist()) if keep else []


def select_rows(spans: list[AbstractSpan], resp_len: int, cfg: AbstractRLConfig, seed: int) -> list[int]:
    """Rows with no abstraction and room to hold one, thinned to the graft fraction."""
    budget = cfg.bootstrap_max_abstract_tokens + 16  # tags + separators
    eligible = [i for i, s in enumerate(spans) if not s.valid and s.response_len + budget <= resp_len]
    return thin_rows(eligible, cfg.bootstrap_graft_fraction, seed)


BOXED_RE = re.compile(r"\\boxed\s*\{[^{}]*\}")


def clean_abstract(text: str) -> str:
    """Drop any boxed answer the phase-2 model slipped into the abstraction."""
    return BOXED_RE.sub("", text).strip()


# Markup that shows up when generation collapses but never in mathematical prose.
SOUP_MARKERS = ("`", "={", "|>", "<|")


def is_degenerate(text: str, min_words: int = 8, min_chars: int = 40, min_unique_ratio: float = 0.35) -> bool:
    """Reject token soup before it gets reinforced.

    ponytail: targeted at the two degeneracy modes actually observed from
    Qwen2.5-Math-1.5B at temperature 1.0 -- a repetition loop, and markup soup
    (``. abstract ({`text field={text``) -- not a general quality classifier.
    Quality is Delta's job; this only stops obvious garbage from being reinforced
    while Delta is still too small to discriminate. Widen it if a new mode shows
    up in ``abstract_rl/graft_rejected_rate``.

    The length floors sit well above a plausible short abstract: a collapsing
    self-distillation loop drives length toward zero (observed: 79 -> 3.7 tokens),
    and a 4-word "abstract" is a target worth refusing even when it is not soup.
    """
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    words = stripped.split()
    if len(words) < min_words:
        return True
    if len(set(words)) / len(words) < min_unique_ratio:  # repetition loop
        return True
    return any(marker in stripped for marker in SOUP_MARKERS)


def find_abstract_text(text: str, open_tag: str, close_tag: str) -> tuple[str, str] | None:
    """``(abstract_text, y_text)`` if a well-formed tag pair exists anywhere in
    ``text``, else ``None``. Unlike ``parsing.py``'s scan, this is unwindowed and
    order-agnostic -- pass-1 output under ``bootstrap_mode=fill_gap`` was never told
    a required position. ``y_text`` is ``text`` with the whole abstract block
    (tags included) removed, so it's ready to be reassembled as ``abstract + Y``
    regardless of where the block originally sat.

    First well-formed pair wins (mirrors ``_parse_leading``'s convention, since the
    target layout is abstract-first): a stray later open tag inside ``y_text`` is
    left alone -- reassembly is a text operation here, not the final training parse
    (``parse_batch(..., leading=True)`` re-parses the reassembled text for that).
    """
    open_i = text.find(open_tag)
    if open_i < 0:
        return None
    close_i = text.find(close_tag, open_i + len(open_tag))
    if close_i < 0:
        return None
    abstract_text = text[open_i + len(open_tag) : close_i]
    y_text = (text[:open_i] + text[close_i + len(close_tag) :]).strip()
    return abstract_text, y_text


def build_prior_batch(
    tokenizer, batch, rows: list[int], max_prompt_length: int
) -> tuple[DataProto | None, list[int]]:
    """One short user turn per row asking only for the abstract, from the question
    alone (``PRIOR_TEMPLATE`` -- no solution in the prompt, unlike ``PHASE2_TEMPLATE``).

    No solution-fitting/truncation is needed: the prompt is question-only, so it
    is never at risk of overflowing ``max_prompt_length`` the way a pasted-in
    solution is.
    """
    from .config import PRIOR_TEMPLATE
    from .scoring import _question_text

    raw = np.empty(len(rows), dtype=object)
    passthrough = {key: np.empty(len(rows), dtype=object) for key in ("data_source", "reward_model", "extra_info")}
    kept = []
    for k, i in enumerate(rows):
        content = PRIOR_TEMPLATE.format(question=_question_text(tokenizer, batch, i))
        if _prompt_tokens(tokenizer, content) > max_prompt_length:
            continue
        raw[len(kept)] = [{"role": "user", "content": content}]
        for key, column in passthrough.items():
            source = batch.non_tensor_batch.get(key)
            column[len(kept)] = source[i] if source is not None else ({} if key != "data_source" else "math_dapo")
        kept.append(i)
    if not kept:
        return None, []
    raw = raw[: len(kept)]
    passthrough = {key: column[: len(kept)] for key, column in passthrough.items()}
    return DataProto.from_single_dict({"raw_prompt": raw, **passthrough}), kept


def reposition_responses(
    tokenizer,
    batch,
    assignments: dict[int, tuple[str, str]],
    cfg: AbstractRLConfig,
) -> tuple[int, int]:
    """Replace ``responses[i]`` wholesale with ``abstract (tagged) + "\\n" + Y``,
    tokenized fresh from the reassembled text. Unlike ``graft_texts`` (append at
    the tail), this **overwrites the entire row**, including zeroing out whatever
    used to occupy the tail past the new length -- the old content is not a valid
    suffix of the reassembled sequence.

    ``assignments`` maps row index -> ``(abstract_text, y_text)``. Returns
    ``(repositioned, rejected)``; rejects degenerate abstracts or a reassembled
    sequence that doesn't fit ``resp_len``.
    """
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    position_ids = batch.batch["position_ids"]
    resp_len = responses.shape[1]
    prompt_len = input_ids.shape[1] - resp_len
    if position_ids.dim() != 2:
        return 0, 0

    repositioned = 0
    rejected = 0
    for i, (abstract_text, y_text) in assignments.items():
        cleaned = clean_abstract(abstract_text)
        if not cleaned or is_degenerate(cleaned):
            rejected += 1
            continue
        full_text = f"{cfg.open_tag}\n{cleaned}\n{cfg.close_tag}\n{y_text}"
        ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if not ids or len(ids) > resp_len:
            rejected += 1
            continue

        new_ids = torch.tensor(ids, dtype=responses.dtype, device=responses.device)
        n = len(ids)
        responses[i, :n] = new_ids
        responses[i, n:] = 0
        response_mask[i, :n] = 1
        response_mask[i, n:] = 0
        input_ids[i, prompt_len : prompt_len + n] = new_ids
        input_ids[i, prompt_len + n :] = 0
        attention_mask[i, prompt_len : prompt_len + n] = 1
        attention_mask[i, prompt_len + n :] = 0
        last_pos = int(position_ids[i, prompt_len - 1]) if prompt_len > 0 else -1
        position_ids[i, prompt_len : prompt_len + n] = torch.arange(
            last_pos + 1, last_pos + 1 + n, dtype=position_ids.dtype, device=position_ids.device
        )
        position_ids[i, prompt_len + n :] = position_ids[i, prompt_len + n - 1] if n > 0 else last_pos
        repositioned += 1

    if repositioned and "rollout_log_probs" in batch.batch:
        batch.batch.pop("rollout_log_probs")
    return repositioned, rejected


class AbstractCache:
    """Question -> abstraction, pre-generated offline by a frozen model.

    The anchor that makes the warm-up stable: these targets never move, so the
    policy cannot chase its own degradation down. Also removes the second
    generation stage from the training loop entirely, since nothing needs the
    rollout's solution any more.

    Built by ``examples/abstract_rl/pregen_abstracts.py``.
    """

    def __init__(self, path: str):
        import pandas as pd

        df = pd.read_parquet(path)
        self.by_question: dict[str, str] = dict(zip(df["question"], df["abstract"], strict=True))
        self.hits = 0
        self.misses = 0

    def get(self, question: str) -> str | None:
        text = self.by_question.get(question)
        if text is None:
            self.misses += 1
        else:
            self.hits += 1
        return text

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _render(question: str, solution: str) -> str:
    return PHASE2_TEMPLATE.format(question=question, solution=solution)


def _prompt_tokens(tokenizer, content: str) -> int:
    return len(
        tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True)
    )


def fit_phase2_prompt(tokenizer, question: str, solution: str, max_prompt_length: int, margin: int = 16) -> str | None:
    """Render the phase-2 prompt, trimming the solution to fit the prompt budget.

    The rollout pads every prompt to ``max_prompt_length`` and errors on any row
    that exceeds it, so a phase-2 prompt carrying a full 3000-token solution takes
    the whole batch down. The solution is trimmed from the tail: the abstraction
    describes the method, which is developed early, and the final answer is
    explicitly not wanted in it anyway.

    Returns ``None`` when even an empty solution does not fit, in which case the
    row cannot be grafted.
    """
    content = _render(question, solution)
    if _prompt_tokens(tokenizer, content) <= max_prompt_length:
        return content

    overhead = _prompt_tokens(tokenizer, _render(question, ""))
    allowed = max_prompt_length - overhead - margin
    if allowed <= 0:
        return None

    ids = tokenizer(solution, add_special_tokens=False)["input_ids"]
    for _ in range(4):  # decode/encode is not exactly length-preserving; converge
        content = _render(question, tokenizer.decode(ids[:allowed]))
        if _prompt_tokens(tokenizer, content) <= max_prompt_length:
            return content
        allowed = int(allowed * 0.8)
        if allowed <= 0:
            return None
    return None


def build_phase2_batch(
    tokenizer, batch, spans: list[AbstractSpan], rows: list[int], max_prompt_length: int
) -> tuple[DataProto | None, list[int]]:
    """One short user turn per row: the question, the model's own solution, and the ask.

    The verifier columns are carried over because the agent loop scores every
    generation unconditionally -- there is no way to ask it not to. The resulting
    score is meaningless (an abstract has no boxed answer) and is never read; it
    costs one cheap verifier call per grafted row.
    """
    from .scoring import _question_text

    responses = batch.batch["responses"]
    contents, kept = [], []
    for i in rows:
        solution = tokenizer.decode(responses[i, : spans[i].response_len], skip_special_tokens=True)
        content = fit_phase2_prompt(tokenizer, _question_text(tokenizer, batch, i), solution, max_prompt_length)
        if content is None:
            continue
        contents.append(content)
        kept.append(i)
    if not kept:
        return None, []

    raw = np.empty(len(kept), dtype=object)
    passthrough = {key: np.empty(len(kept), dtype=object) for key in ("data_source", "reward_model", "extra_info")}
    for k, i in enumerate(kept):
        raw[k] = [{"role": "user", "content": contents[k]}]
        for key, column in passthrough.items():
            source = batch.non_tensor_batch.get(key)
            column[k] = source[i] if source is not None else ({} if key != "data_source" else "math_dapo")
    return DataProto.from_single_dict({"raw_prompt": raw, **passthrough}), kept


def graft(
    tokenizer,
    batch,
    spans: list[AbstractSpan],
    rows: list[int],
    phase2: DataProto,
    cfg: AbstractRLConfig,
) -> tuple[int, int]:
    """Decode online phase-2 output and splice it in. See ``graft_texts``."""
    out_responses = phase2.batch["responses"]
    out_mask = phase2.batch["response_mask"]
    texts = []
    for k in range(len(rows)):
        valid = int(out_mask[k].sum())
        texts.append(tokenizer.decode(out_responses[k, :valid], skip_special_tokens=True) if valid else "")
    return graft_texts(tokenizer, batch, spans, rows, texts, cfg)


def graft_texts(
    tokenizer,
    batch,
    spans: list[AbstractSpan],
    rows: list[int],
    texts: list[str],
    cfg: AbstractRLConfig,
) -> tuple[int, int]:
    """Splice already-decoded abstractions into the rollout responses, in place.

    Shared by the online phase-2 path and the offline ``AbstractCache`` path, so
    both go through identical tokenisation, filtering and tensor patching.

    Returns ``(grafted, rejected)``. Everything the actor consumes is kept
    consistent: responses, response_mask, input_ids, attention_mask and
    position_ids. ``rm_scores`` is left alone -- the advantage sums it over the
    sequence, so the scalar's position does not matter.
    """
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    position_ids = batch.batch["position_ids"]
    resp_len = responses.shape[1]
    prompt_len = input_ids.shape[1] - resp_len
    if position_ids.dim() != 2:
        return 0, 0

    grafted = 0
    rejected = 0

    for k, i in enumerate(rows):
        text = clean_abstract(texts[k])
        if not text or is_degenerate(text):
            rejected += 1
            continue
        block = f"\n{cfg.open_tag}\n{text}\n{cfg.close_tag}"
        ids = tokenizer(block, add_special_tokens=False)["input_ids"]
        ids = ids[: cfg.bootstrap_max_abstract_tokens]
        if not ids:
            continue
        # Re-close the block if the token cap cut the closing tag off.
        if tokenizer.decode(ids).rstrip().endswith(cfg.close_tag) is False:
            tail = tokenizer(f"\n{cfg.close_tag}", add_special_tokens=False)["input_ids"]
            ids = ids[: max(0, cfg.bootstrap_max_abstract_tokens - len(tail))] + tail

        start = spans[i].response_len
        end = start + len(ids)
        if end > resp_len:
            continue

        graft_ids = torch.tensor(ids, dtype=responses.dtype, device=responses.device)
        responses[i, start:end] = graft_ids
        response_mask[i, start:end] = 1
        input_ids[i, prompt_len + start : prompt_len + end] = graft_ids
        attention_mask[i, prompt_len + start : prompt_len + end] = 1
        last_pos = int(position_ids[i, prompt_len + start - 1]) if prompt_len + start > 0 else -1
        position_ids[i, prompt_len + start : prompt_len + end] = torch.arange(
            last_pos + 1, last_pos + 1 + len(ids), dtype=position_ids.dtype, device=position_ids.device
        )
        grafted += 1

    # A grafted row was never sampled token-by-token, so any stored rollout
    # log-probs no longer describe it. Drop them and let the actor recompute.
    if grafted and "rollout_log_probs" in batch.batch:
        batch.batch.pop("rollout_log_probs")
    return grafted, rejected
