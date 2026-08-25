# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""One batched teacher-forced scoring pass for abstraction utility + prior.

Three row classes are built from the *already generated* rollouts -- no new
decoding happens anywhere in this module:

    C  sufficiency  prompt = suffix template with A_i   target = Y_i
    B  baseline     prompt = suffix template, empty A   target = Y_i
    P  prior        prompt = prior template             target = A_i

All rows go through a single ``compute_log_prob`` call on the actor worker
(``forward_only=True``, ``no_grad``), so the scorer is the pre-update policy
snapshot ``pi_bar`` -- the same snapshot ``old_log_probs`` comes from -- and no
gradient can reach it.

The targets are *slices of the rollout token ids*, never re-tokenized text, so
the class-P per-token log-probs align 1:1 with the abstraction positions in the
actor's response frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from .config import PRIOR_TEMPLATE, SUFFICIENCY_TEMPLATE, AbstractRLConfig
from .parsing import AbstractSpan

CLASS_COND = 0  # log pi_bar(Y | X, A)
CLASS_BASE = 1  # log pi_bar(Y | X, empty)
CLASS_PRIOR = 2  # log pi_bar(A | X)


@dataclass
class ScoringResult:
    delta: torch.Tensor  # (B,) matched or stored Delta, nan where not scored
    delta_stored: torch.Tensor  # (B,) spec-exact Delta (rollout-logprob baseline), nan where not scored
    delta_tilde: torch.Tensor  # (B,) clipped, -1 for invalid abstractions
    utility: torch.Tensor  # (B,) U_i = r_i (1 + delta_tilde)
    prior_log_probs: torch.Tensor  # (B, resp_len) prior log-probs at abstraction positions, 0 elsewhere
    scored: torch.Tensor  # (B,) bool, a row actually went through the scorer
    n_rows: int = 0
    n_tokens: int = 0
    debug: list[dict] = field(default_factory=list)


def _question_text(tokenizer, batch, idx: int) -> str:
    """The bare question X, preferring the preprocessed field over a decode."""
    extra = batch.non_tensor_batch.get("extra_info", None)
    if extra is not None:
        problem = extra[idx].get("problem") if isinstance(extra[idx], dict) else None
        if problem:
            return str(problem)
    prompts = batch.batch["prompts"][idx]
    attn = batch.batch["attention_mask"][idx, : prompts.shape[0]].bool()
    return tokenizer.decode(prompts[attn], skip_special_tokens=True)


def _encode_prompt(tokenizer, text: str) -> list[int]:
    """Encode a scoring prompt, wrapped in the chat template when the model has one.

    The spec's prompt text is preserved verbatim; it becomes the user turn so the
    instruct-tuned scorer sees an in-distribution prompt and the target begins
    exactly at the assistant header.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_scoring_batch(
    tokenizer,
    batch,
    spans: list[AbstractSpan],
    rows: list[int],
    cfg: AbstractRLConfig,
) -> tuple[torch.Tensor | None, list[tuple[int, int, int]]]:
    """Build the padded scoring batch.

    Returns ``(tensordict, meta)`` where ``meta[k] = (sample_idx, class, target_len)``
    for scoring row ``k``. Returns ``(None, [])`` when there is nothing to score.
    """
    responses = batch.batch["responses"]
    prompt_ids_list: list[list[int]] = []
    target_ids_list: list[list[int]] = []
    meta: list[tuple[int, int, int]] = []

    want_cond = cfg.use_utility
    want_base = cfg.use_utility and cfg.delta_baseline == "matched"

    for i in rows:
        span = spans[i]
        question = _question_text(tokenizer, batch, i)
        y_ids = responses[i, span.y_start : span.y_end].tolist()
        a_ids = responses[i, span.a_start : span.a_end].tolist()
        if cfg.max_score_response_tokens > 0:
            # ponytail: keeps the scoring forward bounded on pathological traces;
            # Delta is a per-token mean so a suffix cap only reduces its sample size.
            y_ids = y_ids[-cfg.max_score_response_tokens :]
        if not y_ids or not a_ids:
            continue

        if want_cond:
            prompt_ids_list.append(
                _encode_prompt(tokenizer, SUFFICIENCY_TEMPLATE.format(question=question, abstract=span.abstract_text))
            )
            target_ids_list.append(y_ids)
            meta.append((i, CLASS_COND, len(y_ids)))
        if want_base:
            prompt_ids_list.append(
                _encode_prompt(tokenizer, SUFFICIENCY_TEMPLATE.format(question=question, abstract=""))
            )
            target_ids_list.append(y_ids)
            meta.append((i, CLASS_BASE, len(y_ids)))

        prompt_ids_list.append(_encode_prompt(tokenizer, PRIOR_TEMPLATE.format(question=question)))
        target_ids_list.append(a_ids)
        meta.append((i, CLASS_PRIOR, len(a_ids)))

    if not meta:
        return None, []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    max_p = max(len(p) for p in prompt_ids_list)
    max_r = max(len(t) for t in target_ids_list)
    n = len(meta)

    prompts = torch.full((n, max_p), pad_id, dtype=torch.long)
    targets = torch.full((n, max_r), pad_id, dtype=torch.long)
    attn = torch.zeros((n, max_p + max_r), dtype=torch.long)
    resp_mask = torch.zeros((n, max_r), dtype=torch.long)
    for k, (prompt, target) in enumerate(zip(prompt_ids_list, target_ids_list, strict=True)):
        prompts[k, max_p - len(prompt) :] = torch.tensor(prompt, dtype=torch.long)  # left pad
        targets[k, : len(target)] = torch.tensor(target, dtype=torch.long)  # right pad
        attn[k, max_p - len(prompt) : max_p + len(target)] = 1
        resp_mask[k, : len(target)] = 1

    input_ids = torch.cat([prompts, targets], dim=1)
    position_ids = (attn.cumsum(dim=1) - 1).clamp(min=0) * attn

    td = tu.get_tensordict(
        {
            "prompts": prompts,
            "responses": targets,
            "input_ids": input_ids,
            "attention_mask": attn,
            "response_mask": resp_mask,
            "position_ids": position_ids,
        }
    )
    return td, meta


def run_scoring(worker_group, td, cfg: AbstractRLConfig, temperature: float) -> torch.Tensor:
    """One forward-only pass; returns per-token log-probs shaped (n_rows, max_r).

    ``temperature`` must be the rollout temperature: it is what ``old_log_probs``
    was scored at, so using it here keeps Delta's two terms and the compression KL
    on one common scale.
    """
    world_size = getattr(worker_group, "world_size", 1) or 1
    n = td.batch_size[0]
    pad_rows = (-n) % world_size
    if pad_rows:
        # The compute dispatch chunks along the batch dim; duplicate the last row so
        # the scoring batch divides evenly across ranks, then drop the copies.
        td = td[torch.arange(n + pad_rows).clamp(max=n - 1)]

    td = left_right_2_no_padding(td)
    tu.assign_non_tensor(
        td,
        calculate_entropy=False,
        compute_loss=False,
        temperature=temperature,
        max_token_len_per_gpu=cfg.scoring_max_token_len_per_gpu,
    )
    output = worker_group.compute_log_prob(td)
    log_probs = no_padding_2_padding(tu.get(output, "log_probs"), td).float()
    return log_probs[:n]


def compute_delta_utility(
    spans: list[AbstractSpan],
    task_reward: torch.Tensor,
    log_probs: torch.Tensor | None,
    meta: list[tuple[int, int, int]],
    stored_y_log_prob: torch.Tensor,
    cfg: AbstractRLConfig,
    resp_len: int,
) -> ScoringResult:
    """Turn the scoring log-probs into Delta, U and the aligned prior tensor.

    ``stored_y_log_prob`` is the rollout ``old_log_probs`` summed over Y -- the
    spec-exact baseline. It is always used for ``delta_stored`` (logged as a
    diagnostic) and drives the reward when ``delta_baseline='stored'``.
    """
    n_samples = len(spans)
    device = task_reward.device
    nan = float("nan")
    delta = torch.full((n_samples,), nan, device=device)
    delta_stored = torch.full((n_samples,), nan, device=device)
    scored = torch.zeros(n_samples, dtype=torch.bool, device=device)
    prior = torch.zeros((n_samples, resp_len), device=device)

    cond: dict[int, float] = {}
    base: dict[int, float] = {}
    scored_y_len: dict[int, int] = {}
    if log_probs is not None:
        for k, (i, cls, tgt_len) in enumerate(meta):
            row = log_probs[k, :tgt_len]
            if cls == CLASS_COND:
                cond[i] = float(row.sum())
                scored_y_len[i] = tgt_len
            elif cls == CLASS_BASE:
                base[i] = float(row.sum())
            else:
                span = spans[i]
                prior[i, span.a_start : span.a_end] = row.to(device)
                scored[i] = True

    for i, span in enumerate(spans):
        if i not in cond:
            continue
        # Delta is a per-token mean over exactly the tokens that were scored, which
        # differs from |Y| only when max_score_response_tokens caps the row.
        n_y = max(1, scored_y_len[i])
        if n_y == span.n_reasoning:
            # The stored baseline covers all of Y, so it is only comparable uncapped.
            delta_stored[i] = (cond[i] - float(stored_y_log_prob[i])) / n_y
        if cfg.delta_baseline == "matched":
            if i in base:
                delta[i] = (cond[i] - base[i]) / n_y
        else:
            delta[i] = delta_stored[i]

    valid = torch.tensor([s.valid for s in spans], device=device)
    clip = cfg.delta_clip
    # The spec's -clip for an invalid abstraction sends U_i = r_i(1 + Delta) to 0,
    # which erases correctness: a solved problem with no <abstract> scores exactly
    # what a wrong one does. That is only survivable once most rollouts already
    # carry the tag. Measured at valid_abstract_rate ~0.10 it flattened 81-91% of
    # groups, drove pg_loss to 0.001 (GRPO's was 0.0055) and train accuracy fell
    # 0.29 -> 0.07 in nine steps while plain GRPO on the same model/data rose to
    # 0.38. invalid_delta="zero" instead gives U_i = r_i, so an invalid row keeps
    # its full task reward and a valid one still beats it whenever Delta > 0 --
    # format pressure without the death penalty.
    invalid_fill = -clip if cfg.invalid_delta == "penalty" else 0.0
    # valid_bonus is a flat floor so a valid-but-Delta<=0 row still beats an invalid
    # one -- Delta alone was too noisy early on to hold the format up (see the note
    # on valid_bonus in config.py). Reclamped to [-clip, clip] so U stays in [0, 2]
    # (a row already at the clip ceiling is unaffected; a low/negative-Delta row is
    # pulled up toward it).
    delta_tilde = torch.where(
        valid & torch.isfinite(delta),
        (torch.nan_to_num(delta, nan=0.0).clamp(-clip, clip) + cfg.valid_bonus).clamp(-clip, clip),
        torch.full_like(delta, invalid_fill),
    )
    if not cfg.use_utility:
        delta_tilde = torch.zeros_like(delta_tilde)
    utility = task_reward.float() * (1.0 + delta_tilde)

    # Length penalty (plan section 15): a per-row rate cost, beta * |A_i| /
    # max_abstract_tokens, subtracted directly from U on VALID rows only --
    # invalid (incl. too_long) rows already lost delta_tilde/valid_bonus above,
    # so this only needs to bite while a row is still under the hard cap, making
    # the marginal abstraction token cost something even before it trips the cap.
    # Without this the compression KL loss was the only length pressure, and it's
    # a batch-wide mean (loss.py) that dilutes to ~free -- see the fillgap-strong-
    # 20260807-v3 measurement (|A| tripled, KL stayed flat).
    if cfg.length_penalty_coef > 0 and cfg.max_abstract_tokens > 0:
        n_abstract = torch.tensor([float(s.n_abstract) for s in spans], device=device)
        penalty = cfg.length_penalty_coef * (n_abstract / cfg.max_abstract_tokens)
        utility = utility - torch.where(valid, penalty, torch.zeros_like(penalty))

    return ScoringResult(
        delta=delta,
        delta_stored=delta_stored,
        delta_tilde=delta_tilde,
        utility=utility,
        prior_log_probs=prior,
        scored=scored,
        n_rows=len(meta),
        n_tokens=int(sum(m[2] for m in meta)),
    )
