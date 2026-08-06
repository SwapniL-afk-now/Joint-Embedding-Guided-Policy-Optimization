# GPI-CE — Group Policy-Improvement Cross Entropy

A finite-group policy-improvement objective. Not a GRPO loss variant: there is no
PPO ratio, no clipping, no advantage and no critic.

```
q*_b = softmax_i( s^old_b,i + r_b,i / tau )          reward-tilted target
L    = -(1/B) sum_b sum_i q*_b,i log p^theta_b,i     fit the actor to it
dL/ds_i = (p^theta_i - q*_i) / B
```

`s_b,i` is the **response-token mean** log-probability of candidate `i` of prompt
`b` (a sequence sum would let short responses win the group softmax for free).
`q*` is the exact maximizer of `E_q[r] - tau*KL(q || p_old)` over the simplex, so
`E_q*[r] >= E_p_old[r]` always, and constant rewards inside a group give
`q* == p_old` and therefore zero gradient.

## Running

```bash
bash examples/gpi_ce/run_gpi_ce_fsdp.sh                       # 4 online + 4 offline
GPI_ONLINE=8 GPI_OFFLINE=0 bash examples/gpi_ce/run_gpi_ce_fsdp.sh   # online-only control
GPI_ONLINE=2 GPI_OFFLINE=6 bash examples/gpi_ce/run_gpi_ce_fsdp.sh
```

The compute claim to test: `(4,4)` should approach `GRPO(8,0)` while generating
half as many responses. Generation cost drops; **trained** tokens do not, since
offline candidates still go through the backward pass. Report both
`system/generated_tokens` and `system/total_actor_tokens`.

## Training data

`data.custom_cls` points at `verl/experimental/gpi_ce/dataset.py`. Each parquet row
adds two columns to the usual RLHF schema:

```python
{
    "prompt": [...],
    "data_source": "math",
    "reward_model": {"ground_truth": "..."},
    "offline_responses": ["...", "...", "...", "..."],   # DISTINCT verified solutions
    "offline_rewards": [1.0, 1.0, 1.0, 1.0],             # diagnostics only by default
    "extra_info": {"problem_id": "..."},
}
```

The offline responses must be genuine answers **to that prompt** — the group
softmax is only defined over candidates sharing one prompt and one verifier.
Repeating the same trace four times inflates its candidate mass artificially; use
distinct traces or lower `GPI_OFFLINE`.

Offline rows are scored by the same reward path as online rows, so `offline_rewards`
is not trusted at train time (`custom_gpi_ce.verify_offline`, default true).

## Lifecycle

The stock synchronous verl loop already is the required lifecycle — one parameter
sync per rollout batch, no `parameter_sync_step` knob needed:

```
vLLM generate (K_on)  ->  checkpoint_manager.sleep_replicas()
  ->  append K_off offline candidates  ->  reward (same verifier for both sources)
  ->  actor score under theta_n  ->  q*  ->  ONE optimizer step
  ->  checkpoint_manager.update_weights()  [sync + wake]  ->  generate again
```

## Group atomicity

The loss is a softmax across the K candidates of one prompt, so a group may not be
split across a DP rank, a mini-batch or a micro-batch. Three existing mechanisms
are reused rather than reimplemented:

| Level | Mechanism |
|---|---|
| DP rank | `preserve_rollout_groups` -> `get_group_balanced_partitions` |
| mini-batch | `ppo_mini_batch_size == train_batch_size` + `accumulate_minibatch_grads=true` |
| micro-batch | `force_group_size=K` in `prepare_micro_batches` |

`force_group_size` works on **both** micro-batching paths, so `use_dynamic_bsz=true`
is fine and is the default: `rearrange_micro_batches` sums workloads per group and
partitions groups rather than samples, so token-budget batching never cuts a group.
Prefer it whenever candidate lengths are uneven (e.g. short online rollouts beside a
20k-token offline trace) — it bounds micro-batch memory by tokens instead of by a
fixed group count. The only requirement is
`ppo_max_token_len_per_gpu >= max_prompt_length + max_response_length`, which the
config validator enforces because `rearrange_micro_batches` asserts the budget can
hold the longest single sequence.

`compute_gpi_ce_loss` raises rather than silently renormalizing a partial group.

## Key metrics

`gpi/target_reward_improvement` — the guaranteed finite-group improvement; must be
>= 0 every step.

`gpi/target_mass_offline` — **the diagnostic that decides whether mixing works.**
If it stays near zero the actor prior suppresses offline traces before reward
weighting can lift them. Measure `gpi/old_mass_offline` and the
`gpi/seq_score_{online,offline}_mean` gap before adding any new mechanism.

`gpi/projection_kl` — `KL(q* || p_theta)`, how far the actor still is from the
target after the step.

## Tests

```bash
pytest -q tests/experimental/gpi_ce/
```

Covers the closed form of `q*` (`q1/q2 == exp(dr/tau)`), the analytic score
gradient in fp64, token-gradient uniformity and zero padding gradient, permutation
invariance, micro-batch split invariance with a mean-of-means counter-example,
group-split rejection, and the mixed-batch assembly invariants.
