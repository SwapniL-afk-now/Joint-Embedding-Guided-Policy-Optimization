> **OBSOLETE DESIGN NOTE:** This document describes the discarded failure-only auxiliary-loss plan.
> The implemented paper path uses `A_reg = log(pi_anchor/pi_old) + c_x log(pi_old/pi_failure)`,
> RMS-matches `A_reg`, and applies one PPO surrogate to `A_GRPO + beta A_reg`.

# Optimized Implementation Plan

## 1. Final algorithm to implement

Remove the explicit anchor model from the training path. Use the old policy and PPO clipping as the local stability mechanism.

The auxiliary token advantage becomes:

\[
A^{\mathrm{fail}}_{i,t}
=
c_{x_i}
\left[
\log \pi_{\mathrm{old}}(y_{i,t}\mid s_{i,t})
-
\log \pi_{\mathrm{failure}}(y_{i,t}\mid s_{i,t})
\right],
\]

with

\[
c_x = 1-\bar r_x.
\]

Therefore:

| Group | \(c_x\) | Auxiliary behavior |
|---|---:|---|
| All wrong | 1 | Full failure-guided exploration |
| Mixed | Between 0 and 1 | Partial failure avoidance |
| All correct | 0 | No auxiliary update |

Use two separately clipped policy losses:

\[
L_{\mathrm{total}}
=
L_{\mathrm{GRPO}}
+
\beta L_{\mathrm{fail}}.
\]

Do not merge the advantages before PPO clipping.

The complete step is:

```text
rollout + old token log-probabilities
→ reward and group solve rate
→ score active responses under failure model
→ construct detached token advantages
→ actor forward
→ GRPO loss + β × failure policy loss
→ one backward
→ one optimizer step
```

After rollout generation, this requires:

\[
\boxed{
1\text{ failure-model forward}
+
1\text{ actor forward}
+
1\text{ actor backward}
}
\]

The periodic failure-model SFT cost remains separate.

---

# 2. Preserve a legacy mode

Do not immediately delete the existing implementation.

Add:

```python
loss_version: str = "failure_token_adv"
```

Supported values:

```text
"k3_legacy"
"failure_token_adv"
```

This permits direct ablation and protects existing checkpoints.

The new mode must not initialize, score, refresh, or synchronize the anchor model.

---

# 3. Loss implementation

## File

```text
verl/experimental/tafr_grpo/tafr_loss.py
```

Add two new functions.

## 3.1 Construct the dense token advantage

```python
def compute_failure_token_advantage(
    *,
    old_log_prob: torch.Tensor,
    failure_log_prob: torch.Tensor,
    group_reward_mean: torch.Tensor,
    response_mask: torch.Tensor,
    advantage_clip: float,
    failure_model_ready: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Construct a detached failure-aware advantage for each response token."""
```

Expected shapes:

```text
old_log_prob       [batch, response_length]
failure_log_prob   [batch, response_length]
response_mask      [batch, response_length]
group_reward_mean  [batch] or [batch, 1]
```

Implementation:

```python
reward_mean = group_reward_mean.float().clamp(0.0, 1.0)
difficulty = 1.0 - reward_mean

if not failure_model_ready:
    difficulty = torch.zeros_like(difficulty)

difficulty = difficulty.unsqueeze(-1)

failure_signal = old_log_prob.detach() - failure_log_prob.detach()

raw_advantage = difficulty * failure_signal

clipped_advantage = raw_advantage.clamp(
    min=-advantage_clip,
    max=advantage_clip,
)

advantage = clipped_advantage * response_mask
advantage = advantage.detach()
```

Return metrics:

```python
metrics = {
    "difficulty": difficulty.detach(),
    "failure_signal": failure_signal.detach(),
    "raw_advantage": raw_advantage.detach(),
    "advantage": advantage,
}
```

### Important rules

- Do not sum token scores into a response score.
- Do not broadcast one response advantage to all tokens.
- Do not normalize by sequence length before constructing the advantage.
- Do not center the advantage initially.
- Only response-token positions are active.
- All scorer outputs must be detached.

Use:

```python
advantage_clip = 5.0
```

as the initial default.

---

## 3.2 Convert the advantage into a policy loss

```python
def compute_failure_policy_loss(
    *,
    current_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    failure_advantage: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float,
    loss_agg_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a PPO-clipped token policy loss from failure advantages."""
```

Implementation:

```python
log_ratio = current_log_prob - old_log_prob.detach()
ratio = torch.exp(log_ratio)

clipped_ratio = ratio.clamp(
    min=1.0 - clip_ratio,
    max=1.0 + clip_ratio,
)

unclipped_objective = ratio * failure_advantage
clipped_objective = clipped_ratio * failure_advantage

token_loss = -torch.minimum(
    unclipped_objective,
    clipped_objective,
)
```

Aggregate with the same masking and aggregation utility used by the existing GRPO actor loss:

```python
failure_loss = masked_loss_aggregate(
    token_loss,
    response_mask,
    loss_agg_mode,
)
```

Do not introduce a separate length-normalization convention.

---

# 4. Actor-loss integration

## File

```text
verl/workers/engine/fsdp/transformer_impl.py
```

Replace the active \(k3\) anchor/replay section with:

```python
failure_advantage, advantage_stats = compute_failure_token_advantage(
    old_log_prob=batch["old_log_prob"],
    failure_log_prob=batch["tafr_failure_log_prob"],
    group_reward_mean=batch["tafr_group_reward_mean"],
    response_mask=batch["response_mask"],
    advantage_clip=config.failure_advantage_clip,
    failure_model_ready=batch.meta_info["tafr_failure_model_ready"],
)

failure_loss, failure_loss_stats = compute_failure_policy_loss(
    current_log_prob=current_log_prob,
    old_log_prob=batch["old_log_prob"],
    failure_advantage=failure_advantage,
    response_mask=batch["response_mask"],
    clip_ratio=(
        config.failure_clip_ratio
        if config.failure_clip_ratio is not None
        else config.clip_ratio
    ),
    loss_agg_mode=config.loss_agg_mode,
)

policy_loss = grpo_loss + config.beta * failure_loss
```

Then retain the existing single training operation:

```python
optimizer.zero_grad(set_to_none=True)
policy_loss.backward()
optimizer.step()
```

There must be:

- one actor forward;
- one actor backward;
- one optimizer step.

The failure loss must never call `backward()` independently.

---

# 5. Reuse rollout log-probabilities

## Goal

Remove the separate old-policy scoring forward.

During vLLM rollout generation, save the chosen token log-probability for every generated response token.

Store:

```python
batch["old_log_prob"]
```

with shape:

```text
[batch, response_length]
```

## Required validation

Before enabling this optimization by default, compare vLLM rollout log-probabilities with the existing HF recomputation path on several fixed batches.

Test:

```text
mean absolute difference
maximum absolute difference
PPO ratio difference
response-mask alignment
```

Add a temporary config:

```python
reuse_rollout_log_probs: bool = True
verify_rollout_log_probs: bool = False
```

When verification is enabled:

1. collect vLLM log-probabilities;
2. recompute HF old log-probabilities;
3. log their differences;
4. continue using the existing HF values until parity is established.

Pay particular attention to:

- temperature;
- repetition penalties;
- top-\(p\)/top-\(k\);
- BOS/EOS handling;
- prompt/response token offset;
- padded positions.

The PPO behavior log-probability must represent the actual rollout policy.

---

# 6. Compute the group difficulty correctly

## File

```text
verl/trainer/ppo/ray_trainer.py
```

After rewards are calculated, compute the mean reward per prompt group.

Do not rely blindly on reshape ordering. Prefer the existing group identifier or prompt UID.

Conceptually:

```python
group_reward_mean = mean_reward_for_each_prompt_group(rewards, group_ids)
group_reward_mean_per_rollout = broadcast_to_group_members(
    group_reward_mean,
    group_ids,
)

batch["tafr_group_reward_mean"] = group_reward_mean_per_rollout
```

For binary rewards:

```python
difficulty = 1.0 - group_reward_mean
```

Preserve partial rewards if they already lie in \([0,1]\). Do not binarize or round them.

---

# 7. Score only active groups

The failure contribution is zero when:

```python
group_reward_mean == 1
```

Therefore, do not score all-correct groups under the failure model.

Create:

```python
active_group_mask = (
    batch["tafr_group_reward_mean"] < 1.0
) & failure_model_ready
```

Select only active sequences:

```python
active_batch = batch.select(active_group_mask)
```

Run failure-model scoring on `active_batch`.

Scatter the resulting log-probabilities back into the full batch.

For inactive sequences, set:

```python
failure_log_prob = old_log_prob
```

not zero. Then:

```python
old_log_prob - failure_log_prob == 0
```

by construction.

Store:

```python
batch["tafr_failure_log_prob"]
```

This reduces failure-model token processing whenever solved groups are present.

---

# 8. Remove the anchor from the optimized path

In `failure_token_adv` mode:

- do not initialize the anchor clone;
- do not construct anchor EMA checkpoints;
- do not export anchor LoRA;
- do not upload anchor adapter slot 124;
- do not run anchor scoring;
- do not compute anchor metrics;
- do not save anchor state.

PPO clipping now serves as the local trust-region mechanism.

The new method should be described as:

> PPO-constrained, failure-guided token exploration.

It should no longer be described as an anchor/failure dual-KL objective.

Keep the old anchor path only under:

```text
loss_version="k3_legacy"
```

---

# 9. Simplify vLLM scoring

## Relevant files

```text
vllm_async_server.py
vllm_scoring.py
llm_server.py
```

For the new mode:

- use one failure-model LoRA slot;
- remove duplicated anchor/failure scoring requests;
- replace `score_tafr_logprobs_multi` with a single failure-scoring path, or retain the function but call it with one adapter;
- use prefill-only scoring;
- return only response-token log-probabilities;
- ensure the retry token from `max_tokens=1` is never included.

The effective scoring sequence becomes:

```text
active prompt + response
→ failure LoRA prefill
→ response-token log-probabilities
```

No anchor adapter synchronization is needed.

---

# 10. Failure-model training and refresh

Keep the existing pipeline:

```text
collect verifier-confirmed failures
→ response-token SFT
→ update failure model
→ update failure EMA
→ synchronize scorer adapter
```

## Warm start

Do not use the failure advantage before the failure model has received a meaningful update.

Add persistent state:

```python
failure_model_ready: bool
failure_model_update_count: int
```

Default:

```text
failure_model_ready = False
```

Set it to true after:

```text
failure_model_update_count >= 1
```

and the first adapter synchronization succeeds.

Before readiness:

```text
failure advantage = 0
```

The actor runs ordinary GRPO.

## Refresh schedule

Synchronize the failure scorer after every successful failure-model update, not only every disk-checkpoint interval.

For example:

```text
failure SFT every 2 actor steps
failure adapter refresh every 2 actor steps
disk checkpoint every 10 actor steps
```

This separates model freshness from disk I/O.

Persist readiness and update count in:

```text
tafr_state.pt
```

---

# 11. Configuration changes

## File

```text
verl/experimental/tafr_grpo/config.py
```

Add:

```python
loss_version: str = "failure_token_adv"

beta: float = 0.03

failure_advantage_clip: float = 5.0
failure_clip_ratio: float | None = None

reuse_rollout_log_probs: bool = True
verify_rollout_log_probs: bool = False

score_only_active_groups: bool = True

failure_min_updates_before_use: int = 1
```

Deprecate in the new mode:

```text
anchor_beta
anchor model options
anchor LoRA slot
k3 log-ratio clamp
k3 value clamp
anchor refresh settings
```

Keep them only for `k3_legacy`.

Validate:

```python
assert beta >= 0
assert failure_advantage_clip > 0
assert failure_min_updates_before_use >= 1
```

Retain one main auxiliary weight, \(\beta\).

---

# 12. Run-script changes

## File

```text
examples/tafr_grpo/run_tafr_grpo_fsdp.sh
```

Add:

```bash
TAFR_LOSS_VERSION=failure_token_adv
TAFR_BETA=0.03

TAFR_FAILURE_ADVANTAGE_CLIP=5.0
TAFR_REUSE_ROLLOUT_LOGPROBS=true
TAFR_SCORE_ONLY_ACTIVE_GROUPS=true
TAFR_FAILURE_MIN_UPDATES_BEFORE_USE=1
```

Disable the optimized path’s anchor configuration.

Retain the failure-model settings initially:

```text
failure SFT interval = 2
failure EMA decay = 0.9
failure SFT learning rate = 5e-7
```

Do not change failure-model training and actor-loss design simultaneously during the first pilot.

---

# 13. Metrics

Remove misleading KL names from the new mode.

Log:

```text
tafr_adv/failure_loss
tafr_adv/weighted_failure_loss
tafr_adv/total_policy_loss

tafr_adv/group_difficulty_mean
tafr_adv/active_group_fraction
tafr_adv/scored_token_fraction

tafr_adv/failure_signal_mean
tafr_adv/failure_signal_std
tafr_adv/failure_signal_positive_fraction

tafr_adv/advantage_mean
tafr_adv/advantage_std
tafr_adv/advantage_positive_fraction
tafr_adv/advantage_clip_fraction

tafr_adv/ppo_ratio_mean
tafr_adv/ppo_ratio_clip_fraction

tafr_adv/failure_model_ready
tafr_adv/failure_model_update_count
```

Log by group regime:

```text
tafr_adv/all_wrong/advantage_mean
tafr_adv/all_wrong/advantage_positive_fraction
tafr_adv/mixed/advantage_mean
tafr_adv/all_correct/advantage_abs_max
```

For all-correct groups:

```text
advantage_abs_max
```

should be zero within tolerance.

---

# 14. Unit tests

## Advantage algebra

### Equal actor and failure model

```text
old_log_prob == failure_log_prob
```

Expected:

```text
failure advantage == 0
failure loss == 0
```

### All-correct group

```text
group_reward_mean == 1
```

Expected:

```text
difficulty == 0
failure advantage == 0
```

Changing failure log-probabilities must not affect the result.

### All-wrong group

```text
group_reward_mean == 0
```

Expected:

```text
difficulty == 1
```

When:

```text
failure_log_prob > old_log_prob
```

the advantage must be negative.

When:

```text
old_log_prob > failure_log_prob
```

the advantage must be positive.

### Partial group

For:

```text
group_reward_mean == 0.25
```

expected difficulty:

```text
0.75
```

---

## Gradient tests

Verify:

```text
current_log_prob receives gradient
old_log_prob receives no gradient
failure_log_prob receives no gradient
failure_advantage receives no gradient
```

For a positive advantage, gradient descent should increase the selected token’s current log-probability.

For a negative advantage, it should decrease it.

---

## Mask tests

Changing values at:

- prompt tokens;
- padding tokens;
- retry decode token;

must not change:

- failure loss;
- metrics;
- actor gradients.

---

## PPO clipping tests

Test separately:

```text
positive advantage with ratio above 1 + epsilon
negative advantage with ratio below 1 - epsilon
```

Confirm the correct clipped branch is selected.

---

## Selective scoring tests

Create a batch containing:

- all-wrong group;
- mixed group;
- all-correct group.

Assert:

- only the first two groups are sent to the failure scorer;
- all-correct failure log-probabilities are filled from old log-probabilities;
- output ordering matches the original batch.

---

## Optimizer-step test

Instrument the actor optimizer and assert:

```text
optimizer.step() call count == 1
```

per actor update.

---

## Legacy isolation test

In `failure_token_adv` mode, assert that:

```text
anchor scorer call count == 0
anchor adapter sync count == 0
k3 loss call count == 0
```

---

# 15. End-to-end smoke test

Use:

```text
2 prompts
4 rollouts per prompt
32–64 response tokens
3 actor updates
1 failure-model update
```

Verify:

- no NaNs or infinities;
- response-token alignment;
- rollout log-probability parity;
- failure scorer runs only on active groups;
- auxiliary loss is zero before failure-model readiness;
- auxiliary loss activates after the first successful refresh;
- all-correct groups remain inactive;
- exactly one actor optimizer step occurs;
- checkpoint resume preserves `failure_model_ready`;
- legacy mode remains runnable.

---

# 16. Minimal experiment matrix

Run:

| Variant | GRPO | Failure advantage | Selective scoring |
|---|---:|---:|---:|
| GRPO baseline | Yes | No | — |
| Legacy TAFR | Yes | \(k3\) | No |
| New failure-token method | Yes | Yes | No |
| Optimized method | Yes | Yes | Yes |

For the optimized method, sweep only:

\[
\beta\in\{0.01,\;0.03,\;0.1\}.
\]

Keep all other settings fixed initially.

---

# 17. Acceptance criteria

The implementation is considered correct only if:

1. rollout log-probabilities replace the old-policy forward without ratio mismatch;
2. the anchor forward is completely absent in the optimized mode;
3. solved groups are not sent to the failure scorer;
4. the auxiliary advantage is token-specific and detached;
5. GRPO and failure losses use separate clipping decisions;
6. actor training uses one backward and one optimizer step;
7. all-wrong groups receive nonzero auxiliary gradients after warmup;
8. all-correct groups receive zero auxiliary gradients;
9. no persistent increase occurs in KL, response length, or entropy collapse;
10. held-out all-wrong prompts transition to mixed or solved groups faster than GRPO.

## Final optimized execution path

```text
vLLM rollout
  └─ generated tokens
  └─ exact old token log-probabilities

reward evaluation
  └─ group solve rate
  └─ difficulty = 1 − solve rate

select groups with difficulty > 0
  └─ failure-model prefill scoring

actor training forward
  ├─ standard GRPO loss
  ├─ failure token advantage
  └─ separately clipped failure policy loss

total loss
  └─ one backward
  └─ one optimizer step

periodically
  └─ train failure model on verified failures
  └─ refresh failure scorer adapter
```

This is the implementation  should build: **GRPO plus one failure-model scoring pass, with PPO clipping providing stability and a dense token advantage directing exploration away from recurrent failures.**