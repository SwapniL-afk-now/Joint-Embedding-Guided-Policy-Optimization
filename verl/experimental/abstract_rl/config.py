# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

# The instruction exactly as specified.
ROLLOUT_INSTRUCTION = (
    "Think step by step and provide the final answer in \\boxed{...}.\n\n"
    "After completing the solution, write a concise abstract of the general "
    "reasoning strategy inside:\n\n<abstract>\n...\n</abstract>"
)

# Same requirement, stated imperatively. Measured emission rate on
# Qwen2.5-1.5B-Instruct (24 problems x 4 samples, temperature 1.0):
# spec 1% valid, strong 7% valid. On Qwen2.5-Math-1.5B-Instruct both are 0%
# -- see the launch script notes.
ROLLOUT_INSTRUCTION_STRONG = (
    "Think step by step and provide the final answer in \\boxed{...}.\n\n"
    "Then, on a new line after the boxed answer, write a concise abstract of the "
    "general reasoning strategy, wrapped in <abstract> and </abstract> tags. "
    "Describe the reusable method, not the specific numbers or the final answer.\n\n"
    "Your reply must end with the closing </abstract> tag."
)

# Abstract BEFORE the reasoning instead of after. Both "spec" and "strong" put the
# tag after \boxed{}, right where the model reliably stops (see the diagnostic in
# the launch script's header) -- measured 0/25 stopped early, but ~90% never even
# attempted the tag with |Y| well under the response cap, i.e. a hard stop-token
# habit, not a length problem. Putting the tag first sidesteps that stop point
# entirely, at the cost of turning A from a post-hoc trace summary into an
# upfront strategy the model then reasons from -- a real semantic change, not
# just a reordering; see the design note in the launch script.
ROLLOUT_INSTRUCTION_LEADING = (
    "First, write a concise abstract of the general strategy for solving this kind "
    "of problem, wrapped in <abstract> and </abstract> tags. Describe the reusable "
    "method, not the specific numbers.\n\n"
    "Then, after the closing </abstract> tag, think step by step and provide the "
    "final answer in \\boxed{...}."
)

INSTRUCTIONS = {
    "spec": ROLLOUT_INSTRUCTION,
    "strong": ROLLOUT_INSTRUCTION_STRONG,
    "leading": ROLLOUT_INSTRUCTION_LEADING,
}

# Teacher-forced scoring templates. X is the bare question (extra_info.problem when
# present, else the decoded user turn).
SUFFICIENCY_TEMPLATE = (
    "Question: {question}\n\nAbstract strategy:\n{abstract}\n\nReproduce the reasoning and final answer:"
)
PRIOR_TEMPLATE = "Question: {question}\n\nWrite the concise abstract reasoning strategy:"


@dataclass
class AbstractRLConfig:
    """Typed config for abstract-reasoning compression GRPO.

        L_total = L_GRPO(U) + kl_coef * L_compression

    with ``U_i = r_i (1 + clip(Delta_i, -1, 1))`` (``-1`` when the abstraction is
    malformed) and ``L_compression`` the k3 KL between the current policy's
    abstraction distribution and the frozen question-only prior, restricted to
    correct rollouts with a valid abstraction.
    """

    enable: bool = False

    # --- objective ---
    # false -> U_i = r_i (ablation 3: compression only, no usefulness bonus)
    use_utility: bool = True
    # lambda; 0 -> no compression term (ablation 2: usefulness only)
    kl_coef: float = 0.05
    # verl KL estimator name; "low_var_kl" is k3
    kl_type: str = "low_var_kl"
    # Per-token cap on the compression KL, applied on top of k3's own +-10 clamp
    # (core_algos.py, shared by every loss in the repo -- not tightened there).
    # 10 nats is a huge ceiling (exp(10)~22000): a handful of freshly-grafted
    # tokens the policy has never produced in that context sit near it and
    # dominate the batch mean and its gradient. Measured grad_norm ~2-2.5 on
    # abstractrl-twopass-math15b-20260806 (steps 1-3) vs plain GRPO's ~0.2 at the
    # same lr, with abstract_gradient_token_fraction only ~0.14 -- a few outlier
    # tokens, not the bulk of the span. 0 disables (falls back to k3's clamp only).
    kl_clip: float = 2.0
    delta_clip: float = 1.0
    # What Delta_tilde becomes when the abstraction is missing or malformed.
    # "zero"    -> U_i = r_i: the row keeps its task reward, valid rows still win
    #              on Delta. Default; see the note in scoring.py.
    # "penalty" -> U_i = 0 (the spec). Only usable once the format is established.
    invalid_delta: str = "zero"
    # Flat bonus added to Delta_tilde on a *valid* row, on top of the Delta score.
    # invalid_delta="zero" removed the only pressure toward emitting the tag (an
    # invalid row scores the same as a Delta=0 valid one), and Delta itself is too
    # noisy this early to carry the format: valid_abstract_rate fell 0.098 -> 0.006
    # over steps 5-21 of abstractrl-v2-gen15b-20260806, a one-way collapse to zero,
    # not noise around a plateau. This restores a floor without reintroducing the
    # -1 death penalty that erased correctness (see invalid_delta's docstring).
    valid_bonus: float = 0.2
    # "matched": Delta = [log pi(Y|X,A) - log pi(Y|X,empty)] / |Y|, both scored under
    #            the same template so the prompt-format shift cancels.
    # "stored" : the spec-exact form, baseline read from the rollout old_log_probs.
    # Both are always logged; this selects which one drives the reward.
    delta_baseline: str = "matched"
    # true -> feed 2U-1 (identical to the +-1 task reward when Delta == 0)
    utility_affine: bool = True

    # --- parsing ---
    open_tag: str = "<abstract>"
    close_tag: str = "</abstract>"
    # only the last N response tokens are scanned for the tags
    max_scan_tokens: int = 512
    # Hard cap on |A_i|. Over budget -> v_i=0 (failure_reason="too_long"), same as
    # missing/degenerate. Fixes reward hacking measured on abstractrl-fillgap-strong-
    # 20260807-v3 (step 410): with no cap, the model learned to dump the ENTIRE
    # derivation into <abstract> (~420-430 tokens) and leave Y a one-line stub
    # ("The final answer is boxed{27}."), since a solver's own question-only prior
    # already predicts most of a solution's content, so copying it costs little KL.
    # See plan.md section 15. 0 disables the cap.
    max_abstract_tokens: int = 120
    # beta: subtracted from U_i as beta * |A_i| / max_abstract_tokens, BEFORE the
    # affine map. A direct per-row rate cost so the marginal token is never free
    # (unlike the KL loss, which without per-row aggregation dilutes across the
    # batch -- see loss.py). Default 0 -> opt-in explicitly at launch (like
    # kl_coef's sibling knobs), so every existing Delta/utility arithmetic test
    # keeps testing exactly the formula it names without also asserting a length
    # penalty value. Recommended nonzero (e.g. 0.3) whenever max_abstract_tokens
    # is in play, per plan section 15.
    length_penalty_coef: float = 0.0

    # --- rollout / scoring ---
    inject_instruction: bool = True
    # "spec" is the instruction exactly as specified; "strong" states the same
    # requirement imperatively and measures ~7x higher emission rate.
    instruction_variant: str = "spec"
    # free-form override; wins over instruction_variant when non-empty
    instruction_override: str = ""
    # "user_turn" appends to the last user message (original behavior); "system"
    # inserts/extends a dedicated system message instead.
    instruction_placement: str = "user_turn"
    # true -> skip incorrect rollouts in the scoring batch (~2x cheaper, but then
    # delta_mean == delta_correct_mean since incorrect rows have no Delta)
    score_only_correct: bool = False
    scoring_max_token_len_per_gpu: int = 24576
    # per-row cap on the reasoning tokens re-scored (0 = no cap)
    max_score_response_tokens: int = 0

    # --- two-phase format bootstrap (see bootstrap.py) ---
    # "trailing": today's append-to-the-tail graft (PHASE2_TEMPLATE, solution
    #   already known, target sits after Y).
    # "fill_gap": pass-1 already asks for abstract+solution together (see
    #   instruction_placement); if a valid abstract is missing/degenerate anywhere
    #   in the output, pass 2 generates ONLY the abstract from the question alone
    #   (PRIOR_TEMPLATE), and the row is reassembled as abstract-then-Y regardless
    #   of where (or whether) pass 1 put it.
    bootstrap_mode: str = "trailing"
    # graft abstractions for the first N steps; 0 disables the whole mechanism.
    # This is a SECOND generation stage and makes the update a reward-weighted
    # self-distillation rather than a strict policy gradient, so it must switch off.
    bootstrap_steps: int = 0
    # fraction of eligible rollouts to graft. Must stay below 1.0: groups need
    # untagged siblings or the gradient on emitting the tags cancels out.
    bootstrap_graft_fraction: float = 0.5
    # latch grafting off once the native emission rate holds at/above this...
    bootstrap_native_threshold: float = 0.3
    # ...for this many consecutive steps.
    bootstrap_patience: int = 3
    bootstrap_max_abstract_tokens: int = 256
    # For the first N steps, replace the policy loss with a plain SFT loss on the
    # abstraction span of correct, valid rollouts. The bootstrap update is not a
    # policy gradient anyway (ratio == 1 by construction), so doing it explicitly
    # is both honest and far stronger: no mean-centering, no clipping, and it
    # applies to every grafted row rather than the above-average ones. Masked to
    # the abstraction so the solver is left alone. Must be <= bootstrap_steps.
    bootstrap_sft_steps: int = 0
    # Warm-up-only optimizer overrides, restored the moment the warm-up ends.
    # The RL lr (1e-6) is far too small to teach a format: with the NLL grad norm
    # at ~15-20 and clip_grad=1.0, every warm-up step is scaled down ~15x AND
    # taken at the RL step size, so 30 steps move the weights essentially not at
    # all. 0 -> keep the RL value.
    bootstrap_sft_lr: float = 0.0
    bootstrap_sft_clip_grad: float = 0.0
    # Abort the warm-up if train accuracy falls this far below its pre-warm-up
    # baseline -- the warm-up is a self-distillation loop and can eat the solver.
    # 0 disables the guard. 0.10 is ~4 sigma of the 5-step window mean (single-step
    # std is ~0.052); the old 0.05 was under 2 sigma and tripped on noise.
    bootstrap_sft_accuracy_drop: float = 0.10
    # Abort the warm-up if mean abstraction length falls below this fraction of the
    # pre-warm-up length. Length leads, accuracy lags. 0 disables the guard.
    bootstrap_sft_min_length_fraction: float = 0.6
    # Parquet of question -> abstract, pre-generated offline by a FROZEN model
    # (examples/abstract_rl/pregen_abstracts.py). Set this and the warm-up targets
    # stop drifting, and the second generation stage disappears from training.
    # Empty -> fall back to online phase-2 generation by the policy itself, which
    # is a self-distillation loop and collapses; see bootstrap.py.
    bootstrap_abstract_cache: str = ""

    # --- debug ---
    debug_print_n: int = 0

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> AbstractRLConfig:
        if not config:
            return cls()
        merged = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)


def rollout_instruction(cfg: AbstractRLConfig) -> str:
    return cfg.instruction_override or INSTRUCTIONS[cfg.instruction_variant]


def apply_instruction(messages: list[dict], instruction: str, placement: str = "user_turn") -> list[dict]:
    """Insert ``instruction`` into ``messages``, idempotently. Shared by the dataset
    (train/val, before prompt-length filtering) and the trainer-side gen-batch hook.

    "system": extend an existing leading system message, or prepend a new one.
    "user_turn": append to the last user message (original behavior).
    """
    out = [dict(m) for m in messages]
    if placement == "system":
        if out and out[0].get("role") == "system":
            if instruction not in out[0]["content"]:
                out[0] = {**out[0], "content": f"{out[0]['content']}\n\n{instruction}"}
            return out
        return [{"role": "system", "content": instruction}, *out]
    for msg in reversed(out):
        if msg.get("role") == "user":
            if instruction not in msg["content"]:
                msg["content"] = f"{msg['content']}\n\n{instruction}"
            break
    return out


def abstract_rl_enabled(config: DictConfig | dict) -> bool:
    custom = config.get("custom_abstract_rl", {}) if config is not None else {}
    return bool(custom and custom.get("enable", False))


def validate_abstract_rl_config(config: DictConfig | dict) -> AbstractRLConfig:
    """Validate the root verl config and return the typed abstract-RL config."""

    custom = AbstractRLConfig.from_config(config.get("custom_abstract_rl", {}) if config is not None else {})
    if not custom.enable:
        return custom

    algorithm = config.get("algorithm", {})
    raw_adv_estimator = algorithm.get("adv_estimator", "")
    adv_estimator = str(getattr(raw_adv_estimator, "value", raw_adv_estimator)).lower()
    if adv_estimator != "grpo":
        raise ValueError("custom_abstract_rl.enable=true requires algorithm.adv_estimator='grpo'.")

    if custom.kl_coef < 0:
        raise ValueError("custom_abstract_rl.kl_coef must be non-negative.")
    if custom.delta_clip <= 0:
        raise ValueError("custom_abstract_rl.delta_clip must be positive.")
    if custom.kl_clip < 0:
        raise ValueError("custom_abstract_rl.kl_clip must be non-negative (0 disables it).")
    if custom.max_scan_tokens <= 0:
        raise ValueError("custom_abstract_rl.max_scan_tokens must be positive.")
    if custom.max_abstract_tokens < 0:
        raise ValueError("custom_abstract_rl.max_abstract_tokens must be non-negative (0 disables the cap).")
    if custom.length_penalty_coef < 0:
        raise ValueError("custom_abstract_rl.length_penalty_coef must be non-negative.")
    if custom.delta_baseline not in ("matched", "stored"):
        raise ValueError(
            f"custom_abstract_rl.delta_baseline must be 'matched' or 'stored', got {custom.delta_baseline!r}."
        )
    if not custom.open_tag or not custom.close_tag:
        raise ValueError("custom_abstract_rl.open_tag / close_tag must be non-empty.")
    if custom.instruction_variant not in INSTRUCTIONS:
        raise ValueError(
            f"custom_abstract_rl.instruction_variant must be one of {sorted(INSTRUCTIONS)}, "
            f"got {custom.instruction_variant!r}."
        )
    if custom.instruction_placement not in ("user_turn", "system"):
        raise ValueError(
            f"custom_abstract_rl.instruction_placement must be 'user_turn' or 'system', "
            f"got {custom.instruction_placement!r}."
        )
    if custom.bootstrap_mode not in ("trailing", "fill_gap"):
        raise ValueError(
            f"custom_abstract_rl.bootstrap_mode must be 'trailing' or 'fill_gap', got {custom.bootstrap_mode!r}."
        )
    if custom.invalid_delta not in ("zero", "penalty"):
        raise ValueError(f"custom_abstract_rl.invalid_delta must be 'zero' or 'penalty', got {custom.invalid_delta!r}.")
    instruction = rollout_instruction(custom)
    if custom.inject_instruction and (custom.open_tag not in instruction or custom.close_tag not in instruction):
        raise ValueError("The rollout instruction must show both abstraction tags, or the policy cannot emit them.")

    if custom.inject_instruction:
        # The instruction must be in the prompt BEFORE maybe_filter_out_long_prompts
        # measures it. Injecting only at _get_gen_batch pushes near-ceiling prompts
        # past max_prompt_length and kills the rollout tens of steps in with
        # "Sizes of tensors must match except in dimension 0".
        data = config.get("data", {}) if config is not None else {}
        custom_cls = (data.get("custom_cls", {}) or {}).get("name")
        if custom_cls != "AbstractRLDataset":
            raise ValueError(
                "custom_abstract_rl.inject_instruction=true requires the instruction to be added "
                "before prompt-length filtering. Set "
                "data.custom_cls.path=verl/experimental/abstract_rl/dataset.py "
                "data.custom_cls.name=AbstractRLDataset "
                f"+data.abstract_rl_instruction={custom.instruction_variant}"
            )
        variant = data.get("abstract_rl_instruction", "spec")
        if variant != custom.instruction_variant:
            raise ValueError(
                f"data.abstract_rl_instruction={variant!r} must match "
                f"custom_abstract_rl.instruction_variant={custom.instruction_variant!r}, "
                "or two different instructions get appended to the same prompt."
            )

    # The reward must be verified on Y only; without the wrapper a \boxed{} inside
    # the abstraction would silently become the graded answer.
    reward_fn = config.get("reward", {}).get("custom_reward_function", {}) or {}
    if not reward_fn.get("path"):
        raise ValueError(
            "custom_abstract_rl.enable=true requires the abstraction-stripping verifier. Set "
            "reward.custom_reward_function.path=verl/experimental/abstract_rl/reward_fn.py "
            "reward.custom_reward_function.name=compute_score_ignoring_abstract"
        )

    if custom.bootstrap_steps > 0:
        # The <1.0 requirement only matters when grafting is teaching the policy to
        # emit the tag *natively* in one pass (inject_instruction=True): held-back
        # rows are what gives GRPO a presence contrast. In two-pass mode
        # (inject_instruction=False) there is no single-pass emission to teach --
        # every row is deterministically grafted every step by construction, so
        # 1.0 is the correct, intended value; GRPO only has quality left to learn.
        if custom.inject_instruction and not 0.0 < custom.bootstrap_graft_fraction < 1.0:
            raise ValueError(
                "custom_abstract_rl.bootstrap_graft_fraction must be strictly between 0 and 1. "
                "At 1.0 every rollout in a group carries an abstraction, the tags appear in "
                "positive- and negative-advantage sequences alike, and the gradient on emitting "
                "them cancels -- the policy would learn abstraction quality but never presence. "
                "(Not applicable when inject_instruction=false -- two-pass mode never needs the "
                "policy to learn native emission.)"
            )
        if not custom.inject_instruction and not 0.0 < custom.bootstrap_graft_fraction <= 1.0:
            raise ValueError("custom_abstract_rl.bootstrap_graft_fraction must be in (0, 1].")
        if not 0.0 < custom.bootstrap_native_threshold <= 1.0:
            raise ValueError("custom_abstract_rl.bootstrap_native_threshold must be in (0, 1].")
        if custom.bootstrap_patience < 1:
            raise ValueError("custom_abstract_rl.bootstrap_patience must be >= 1.")
        if custom.bootstrap_max_abstract_tokens < 8:
            raise ValueError("custom_abstract_rl.bootstrap_max_abstract_tokens must be >= 8.")
        if custom.bootstrap_abstract_cache:
            import os

            if not os.path.exists(custom.bootstrap_abstract_cache):
                raise ValueError(
                    f"custom_abstract_rl.bootstrap_abstract_cache={custom.bootstrap_abstract_cache!r} does not "
                    "exist. Build it with examples/abstract_rl/pregen_abstracts.py."
                )
        elif custom.bootstrap_sft_steps > 0:
            print(
                "WARNING: abstract_rl SFT warm-up with no bootstrap_abstract_cache -- the targets will be "
                "generated by the policy being trained, which is a self-distillation loop. Measured collapse: "
                "mean abstraction length 80 -> 42 tokens and train accuracy 0.55 -> 0.48 within five steps at "
                "lr 3e-6. The length/accuracy guards will abort the warm-up, but prefer a frozen cache."
            )
        if custom.bootstrap_sft_lr < 0 or custom.bootstrap_sft_clip_grad < 0:
            raise ValueError("custom_abstract_rl.bootstrap_sft_lr / _clip_grad must be non-negative.")
        if custom.bootstrap_sft_steps > custom.bootstrap_steps:
            raise ValueError(
                f"custom_abstract_rl.bootstrap_sft_steps ({custom.bootstrap_sft_steps}) must be <= "
                f"bootstrap_steps ({custom.bootstrap_steps}): the SFT warm-up imitates grafted "
                "abstractions, so grafting has to still be running."
            )
        print(
            f"NOTICE: abstract_rl format bootstrap ON for the first {custom.bootstrap_steps} steps "
            f"(graft_fraction={custom.bootstrap_graft_fraction}). This adds a second generation "
            "stage and makes the update a reward-weighted self-distillation until it latches off; "
            "watch abstract_rl/native_abstract_rate and abstract_rl/bootstrap_active."
        )

    if custom.score_only_correct and custom.delta_baseline == "matched":
        print(
            "NOTICE: custom_abstract_rl.score_only_correct=true -> abstract_rl/delta_mean "
            "is computed over correct rollouts only (incorrect rows carry no Delta)."
        )

    return custom
