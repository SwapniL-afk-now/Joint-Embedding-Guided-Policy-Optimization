# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Driver-side hooks: prompt injection, scoring + reward shaping, metrics.

All of it is gated on ``trainer.abstract_rl_enabled``; nothing here runs and no
new batch key exists when the mode is off.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.profiler import marked_timer

from .config import AbstractRLConfig, rollout_instruction
from .parsing import build_masks, parse_batch
from .scoring import build_scoring_batch, compute_delta_utility, run_scoring

EPS = 1e-8


def config_dict(cfg: AbstractRLConfig, global_steps: int = 0, bootstrap=None) -> dict:
    """The subset the actor loss needs, stamped as a non-tensor on the actor batch."""
    return {
        "enable": bool(cfg.enable),
        "kl_coef": float(cfg.kl_coef),
        "kl_type": str(cfg.kl_type),
        "kl_clip": float(cfg.kl_clip),
        # During the format warm-up the policy loss is replaced by a masked SFT
        # loss and the compression KL is off -- pulling the abstraction toward the
        # question-only prior directly fights format acquisition.
        # bootstrap_sft_steps is a floor, not a deadline: ending the warm-up while
        # the policy still emits no abstraction natively drops straight back into
        # v_i=0 -> U_i=0 -> flat groups -> grad_norm 0, which is exactly what
        # abstractrl-warm30-20260806 did for its last 100 steps. So keep warming up
        # until the native-emission latch flips, capped by bootstrap_steps (past
        # which grafting stops and there are no SFT targets anyway).
        "sft_warmup": bool(
            cfg.bootstrap_sft_steps > 0
            and (global_steps <= cfg.bootstrap_sft_steps or not getattr(bootstrap, "disabled", True))
            and global_steps <= cfg.bootstrap_steps
            and not getattr(bootstrap, "sft_disabled", False)
        ),
        "sft_lr": float(cfg.bootstrap_sft_lr),
        "sft_clip_grad": float(cfg.bootstrap_sft_clip_grad),
    }


@contextmanager
def sft_warmup_optimizer(engine, abstract_config):
    """Swap in the warm-up lr / grad clip for the duration of one actor update.

    The warm-up is supervised learning, not RL, and wants a supervised step size.
    Restored on exit (and on exception) so step ``bootstrap_sft_steps + 1`` is
    plain GRPO at the RL settings again. Adam's moment estimates carry over --
    that is deliberate; a fresh optimizer would throw away the solver's state too.
    """
    if not (abstract_config or {}).get("sft_warmup", False):
        yield
        return
    lr = float(abstract_config.get("sft_lr", 0.0) or 0.0)
    clip = float(abstract_config.get("sft_clip_grad", 0.0) or 0.0)
    optimizer = getattr(engine, "optimizer", None)
    opt_config = getattr(engine, "optimizer_config", None)
    saved_lrs = None
    saved_clip = None
    if lr > 0 and optimizer is not None:
        saved_lrs = [g["lr"] for g in optimizer.param_groups]
        for g in optimizer.param_groups:
            g["lr"] = lr
    if clip > 0 and opt_config is not None:
        saved_clip = opt_config.clip_grad
        opt_config.clip_grad = clip
    try:
        yield
    finally:
        if saved_lrs is not None:
            for g, old in zip(optimizer.param_groups, saved_lrs, strict=True):
                g["lr"] = old
        if saved_clip is not None:
            opt_config.clip_grad = saved_clip


def inject_instruction(cfg: AbstractRLConfig, gen_batch) -> None:
    """Append the abstraction instruction to the last user turn, in place.

    Applied in ``_get_gen_batch``, which serves both training and validation, so
    the policy is evaluated on the prompt format it was trained on.
    """
    if not cfg.inject_instruction:
        return
    raw_prompt = gen_batch.non_tensor_batch.get("raw_prompt", None)
    if raw_prompt is None:
        return
    instruction = rollout_instruction(cfg)
    out = np.empty(len(raw_prompt), dtype=object)
    for i, messages in enumerate(raw_prompt):
        messages = [dict(m) for m in messages]
        for msg in reversed(messages):
            if msg.get("role") == "user":
                if instruction not in msg["content"]:
                    msg["content"] = f"{msg['content']}\n\n{instruction}"
                break
        out[i] = messages
    gen_batch.non_tensor_batch["raw_prompt"] = out


def bootstrap_graft(trainer, batch, timing_raw: dict) -> dict[str, float]:
    """Second-phase abstraction generation for rollouts that produced none.

    Runs after the reward (which is computed on Y only, so it is unaffected) and
    before ``old_log_prob``, so the recomputed log-probs cover the grafted tokens.
    Off unless ``bootstrap_steps > 0``; latches off once the policy emits
    abstractions natively. See bootstrap.py for why only a fraction is grafted.
    """
    from .bootstrap import build_phase2_batch, graft, select_rows

    cfg: AbstractRLConfig = trainer.abstract_rl_config
    state = trainer.abstract_rl_bootstrap
    tokenizer = trainer.tokenizer
    responses = batch.batch["responses"]
    n, resp_len = responses.shape

    spans = parse_batch(
        tokenizer,
        responses,
        batch.batch["response_mask"],
        cfg.open_tag,
        cfg.close_tag,
        cfg.max_scan_tokens,
        leading=cfg.instruction_variant == "leading",
    )
    native_rate = sum(s.valid for s in spans) / max(n, 1)
    state.observe(native_rate)

    metrics = {
        "abstract_rl/native_abstract_rate": native_rate,
        "abstract_rl/bootstrap_active": 0.0,
        "abstract_rl/grafted_rate": 0.0,
        "abstract_rl/graft_no_room_rate": 0.0,
    }
    if not state.active(trainer.global_steps):
        return metrics

    metrics["abstract_rl/bootstrap_active"] = 1.0
    no_room = sum(
        1 for s in spans if not s.valid and s.response_len + cfg.bootstrap_max_abstract_tokens + 16 > resp_len
    )
    metrics["abstract_rl/graft_no_room_rate"] = no_room / max(n, 1)

    rows = select_rows(spans, resp_len, cfg, seed=trainer.global_steps)
    if not rows:
        return metrics

    # Offline cache: frozen targets, and no second generation stage at all.
    if cfg.bootstrap_abstract_cache:
        from .bootstrap import AbstractCache, graft_texts
        from .scoring import _question_text

        cache = getattr(trainer, "_abstract_rl_cache", None)
        if cache is None:
            cache = AbstractCache(cfg.bootstrap_abstract_cache)
            trainer._abstract_rl_cache = cache
        with marked_timer("abstract_bootstrap", timing_raw, color="magenta"):
            texts = [cache.get(_question_text(tokenizer, batch, i)) or "" for i in rows]
            grafted, rejected = graft_texts(tokenizer, batch, spans, rows, texts, cfg)
        metrics["abstract_rl/grafted_rate"] = grafted / max(n, 1)
        metrics["abstract_rl/graft_rejected_rate"] = rejected / max(n, 1)
        metrics["abstract_rl/graft_cache_hit_rate"] = cache.hit_rate
        return metrics

    with marked_timer("abstract_bootstrap", timing_raw, color="magenta"):
        gen_batch, rows = build_phase2_batch(tokenizer, batch, spans, rows, int(trainer.config.data.max_prompt_length))
        if gen_batch is None:
            return metrics
        gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "global_steps": trainer.global_steps,
            # Greedy: at the rollout temperature the phase-2 abstractions collapse
            # into token soup, and the sequence gets reinforced whole.
            "validate": True,
        }
        size_divisor = max(1, trainer.config.actor_rollout_ref.rollout.agent.num_workers)
        padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
        phase2 = trainer.async_rollout_manager.generate_sequences(padded)
        phase2 = unpad_dataproto(phase2, pad_size=pad_size)
        grafted, rejected = graft(tokenizer, batch, spans, rows, phase2, cfg)

    metrics["abstract_rl/grafted_rate"] = grafted / max(n, 1)
    metrics["abstract_rl/graft_rejected_rate"] = rejected / max(n, 1)
    return metrics


def _task_reward(batch) -> torch.Tensor:
    """r_i in {0,1}. Prefers the verifier's ``acc`` over the sign of the score,
    which is unsafe under the dapo overlong-buffer penalty."""
    acc = batch.non_tensor_batch.get("acc", None)
    if acc is not None:
        return torch.tensor(np.asarray(acc, dtype=np.float32) > 0.5, dtype=torch.float32)
    return (batch.batch["token_level_scores"].sum(dim=-1) > 0).float()


def _group_metrics(uids: np.ndarray, reward: np.ndarray, utility: np.ndarray) -> dict[str, float]:
    mixed, zero_adv, n = 0, 0, 0
    for uid in np.unique(uids):
        sel = uids == uid
        n += 1
        r = reward[sel]
        if 0 < r.sum() < r.size:
            mixed += 1
        if float(np.std(utility[sel])) < 1e-8:
            zero_adv += 1
    n = max(n, 1)
    return {
        "abstract_rl/mixed_reward_group_fraction": mixed / n,
        "abstract_rl/zero_advantage_group_fraction": zero_adv / n,
    }


def _nanstat(values: torch.Tensor, fn) -> float:
    finite = values[torch.isfinite(values)]
    return float(fn(finite)) if finite.numel() else float("nan")


def score_and_shape(trainer, batch, timing_raw: dict) -> dict[str, float]:
    """Parse, score, and replace the task reward with the utility reward.

    Runs after the reward is materialised and before ``compute_advantage``, so the
    group-relative advantage is computed on U. Writes ``abstract_mask``,
    ``abstract_prior_log_probs`` and ``abstract_kl_weight`` for the actor loss.
    """
    cfg: AbstractRLConfig = trainer.abstract_rl_config
    tokenizer = trainer.tokenizer
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    n, resp_len = responses.shape

    spans = parse_batch(
        tokenizer,
        responses,
        response_mask,
        cfg.open_tag,
        cfg.close_tag,
        cfg.max_scan_tokens,
        leading=cfg.instruction_variant == "leading",
    )
    reasoning_mask, abstract_mask = build_masks(spans, response_mask)
    reward = _task_reward(batch)
    # Runs before _update_actor stamps config_dict, so an abort takes effect this step.
    bootstrap = getattr(trainer, "abstract_rl_bootstrap", None)
    if bootstrap is not None:
        bootstrap.observe_accuracy(float(reward.mean()))
    valid = torch.tensor([s.valid for s in spans], dtype=torch.float32)

    old_log_probs = batch.batch.get("old_log_probs", None)
    if old_log_probs is None:
        # Fused-old-log-prob path: the spec-exact baseline is unavailable, so the
        # matched baseline is the only option (validated in the trainer).
        stored_y_log_prob = torch.full((n,), float("nan"))
    else:
        stored_y_log_prob = (old_log_probs.float() * reasoning_mask.float()).sum(dim=-1)

    rows = [i for i in range(n) if spans[i].valid and (not cfg.score_only_correct or reward[i] > 0)]
    temperature = float(batch.meta_info.get("temperature", trainer.config.actor_rollout_ref.rollout.temperature))
    with marked_timer("abstract_score", timing_raw, color="purple"):
        td, meta = build_scoring_batch(tokenizer, batch, spans, rows, cfg)
        log_probs = run_scoring(trainer.actor_rollout_wg, td, cfg, temperature) if td is not None else None
    result = compute_delta_utility(spans, reward, log_probs, meta, stored_y_log_prob, cfg, resp_len)

    # --- reward shaping: U_i replaces r_i at the last response token ---
    # token_level_rewards aliases token_level_scores; clone before writing or the
    # raw task reward (and the reward tensor the caller holds) is corrupted.
    tlr = batch.batch["token_level_rewards"].clone()
    shaped = 2.0 * result.utility - 1.0 if cfg.utility_affine else result.utility
    device = tlr.device
    col = torch.arange(resp_len, device=device).unsqueeze(0).expand(n, resp_len)
    last_idx = torch.where(response_mask.bool(), col, torch.full_like(col, -1)).max(dim=1).values
    sel = (last_idx >= 0).nonzero(as_tuple=True)[0]
    # Overwrite the scalar the verifier scattered there; any other token-level
    # component (e.g. in-reward KL) is left untouched.
    tlr[sel, last_idx[sel]] = shaped.to(device=device, dtype=tlr.dtype)[sel]
    batch.batch["token_level_rewards"] = tlr

    # --- fields for the actor loss ---
    batch.batch["abstract_mask"] = abstract_mask.to(device=device, dtype=torch.int32)
    batch.batch["abstract_reasoning_mask"] = reasoning_mask.to(device=device, dtype=torch.int32)
    batch.batch["abstract_prior_log_probs"] = result.prior_log_probs.to(device=device, dtype=torch.float32)
    batch.batch["abstract_kl_weight"] = (reward * valid).to(device=device, dtype=torch.float32)

    metrics = _rollout_metrics(spans, reward, valid, abstract_mask, reasoning_mask, result)
    metrics.update(_group_metrics(np.asarray(batch.non_tensor_batch["uid"]), reward.numpy(), result.utility.numpy()))
    metrics.update(_kl_metrics(old_log_probs, result.prior_log_probs, abstract_mask, reward, cfg))
    # Length over VALID rows only -- the batch-wide mean conflates "abstracts got
    # shorter" with "fewer rows carry one", and only the former is the collapse.
    if bootstrap is not None:
        valid_lens = abstract_mask.float().sum(dim=-1)[valid.bool()]
        if valid_lens.numel():
            bootstrap.observe_abstract_length(float(valid_lens.mean()))
        metrics["abstract_rl/sft_aborted"] = float(bootstrap.sft_disabled)
    metrics["abstract_rl/scoring_rows"] = float(result.n_rows)
    metrics["abstract_rl/scoring_target_tokens"] = float(result.n_tokens)
    trainer._abstract_rl_scoring_tokens = float(result.n_tokens)

    if cfg.debug_print_n > 0:
        _print_examples(trainer, batch, spans, reward, result, metrics, cfg)
    return metrics


def _rollout_metrics(spans, reward, valid, abstract_mask, reasoning_mask, result) -> dict[str, float]:
    n = max(len(spans), 1)
    n_abs = abstract_mask.float().sum(dim=-1)
    n_rea = reasoning_mask.float().sum(dim=-1)
    correct = reward > 0
    delta_correct = result.delta[correct] if correct.any() else result.delta[:0]
    finite = result.delta[torch.isfinite(result.delta)]
    return {
        "abstract_rl/task_reward_mean": float(reward.mean()),
        "abstract_rl/pass_rate": float((reward > 0).float().mean()),
        "abstract_rl/valid_abstract_rate": float(valid.mean()),
        "abstract_rl/missing_open_tag_rate": sum(s.missing_open for s in spans) / n,
        "abstract_rl/missing_close_tag_rate": sum(s.missing_close for s in spans) / n,
        "abstract_rl/reversed_tag_rate": sum(s.reversed_tags for s in spans) / n,
        "abstract_rl/empty_abstract_rate": sum(s.empty_abstract for s in spans) / n,
        "abstract_rl/abstract_token_count_mean": float(n_abs.mean()),
        "abstract_rl/reasoning_token_count_mean": float(n_rea.mean()),
        "abstract_rl/compression_ratio": float((n_abs / n_rea.clamp(min=1.0)).mean()),
        "abstract_rl/delta_mean": _nanstat(result.delta, torch.mean),
        "abstract_rl/delta_median": _nanstat(result.delta, torch.median),
        "abstract_rl/delta_correct_mean": _nanstat(delta_correct, torch.mean),
        "abstract_rl/delta_positive_fraction": (float((finite > 0).float().mean()) if finite.numel() else float("nan")),
        "abstract_rl/delta_clipped_fraction": (
            float((finite.abs() > 1.0).float().mean()) if finite.numel() else float("nan")
        ),
        "abstract_rl/delta_stored_mean": _nanstat(result.delta_stored, torch.mean),
        "abstract_rl/delta_format_shift_mean": _nanstat(result.delta_stored - result.delta, torch.mean),
        "abstract_rl/utility_reward_mean": float(result.utility.mean()),
        "abstract_rl/utility_reward_std": float(result.utility.std(unbiased=False)),
    }


def _kl_metrics(old_log_probs, prior, abstract_mask, reward, cfg) -> dict[str, float]:
    """Abstraction KL at the start of the update, where pi_theta == pi_bar exactly.

    The differentiable term is computed in the actor loss; this is the same
    quantity evaluated on the pre-update weights, which is where the distribution
    diagnostics (percentiles) are cheap to take.
    """
    if old_log_probs is None:
        return {}
    mask = abstract_mask.bool()
    if not mask.any():
        return {
            "abstract_rl/abstract_kl_mean": 0.0,
            "abstract_rl/abstract_kl_p50": 0.0,
            "abstract_rl/abstract_kl_p95": 0.0,
        }
    kld = kl_penalty(logprob=old_log_probs.float(), ref_logprob=prior.float(), kl_penalty=cfg.kl_type)
    vals = kld[mask]
    correct_mask = mask & (reward > 0).unsqueeze(-1)
    out = {
        "abstract_rl/abstract_kl_mean": float(vals.mean()),
        "abstract_rl/abstract_kl_p50": float(vals.quantile(0.5)),
        "abstract_rl/abstract_kl_p95": float(vals.quantile(0.95)),
    }
    out["abstract_rl/abstract_kl_correct_mean"] = float(kld[correct_mask].mean()) if correct_mask.any() else 0.0
    return out


def efficiency_metrics(trainer, timing_raw: dict) -> dict[str, float]:
    gen = float(timing_raw.get("gen", 0.0))
    score = float(timing_raw.get("abstract_score", 0.0))
    update = float(timing_raw.get("update_actor", 0.0))
    step = float(timing_raw.get("step", gen + score + update))
    tokens = float(getattr(trainer, "_abstract_rl_scoring_tokens", 0.0))
    metrics = {
        "abstract_rl/rollout_time": gen,
        "abstract_rl/scoring_forward_time": score,
        "abstract_rl/policy_update_time": update,
        "abstract_rl/step_total_time": step,
        "abstract_rl/scoring_overhead_fraction": score / max(step, EPS),
        "abstract_rl/scoring_tokens_per_second": tokens / max(score, EPS),
    }
    if torch.cuda.is_available():
        metrics["abstract_rl/peak_memory"] = torch.cuda.max_memory_allocated() / (1024**3)
    return metrics


def _print_examples(trainer, batch, spans, reward, result, metrics, cfg) -> None:
    tokenizer = trainer.tokenizer
    responses = batch.batch["responses"]
    # Valid abstractions first -- early in training they are the rare case, and
    # they are the ones worth eyeballing.
    order = sorted(range(len(spans)), key=lambda i: (not spans[i].valid, -float(reward[i])))
    for i in order[: cfg.debug_print_n]:
        span = spans[i]
        question = batch.non_tensor_batch.get("extra_info", [{}] * len(spans))[i]
        question = question.get("problem", "<no extra_info.problem>") if isinstance(question, dict) else "?"
        reasoning = tokenizer.decode(responses[i, span.y_start : span.y_end], skip_special_tokens=True)
        print("=" * 88)
        print(f"[abstract_rl] example {i}  valid={span.valid}  |Y|={span.n_reasoning}  |A|={span.n_abstract}")
        print(f"QUESTION   : {str(question)[:400]}")
        print(f"REASONING  : {reasoning[-800:]}")
        print(f"ABSTRACT   : {span.abstract_text[:800]}")
        print(
            f"r={float(reward[i]):.0f}  delta={float(result.delta[i]):.4f}  "
            f"delta_tilde={float(result.delta_tilde[i]):.4f}  U={float(result.utility[i]):.4f}  "
            f"batch_abstract_kl_mean={metrics.get('abstract_rl/abstract_kl_mean', float('nan')):.4f}"
        )
