# SDC Implementation Details

This document describes Success/Failure Distribution Contrast (SDC) in this
repository. SDC is an additive auxiliary objective and can coexist with GRPO,
Dr.GRPO, DAPO, NGRPO, and AVSPO.

## Source map

- `verl/experimental/sdc/config.py`: configuration and validation.
- `verl/experimental/sdc/data_collector.py`: semantic outcomes and buffers.
- `verl/experimental/sdc/sdc_loss.py`: SDC actor-loss algebra.
- `verl/experimental/sdc/sdc_state.py`: frozen-base state and interpolation.
- `verl/experimental/sdc/sft_trainer.py`: teacher SFT and record pairing.
- `verl/experimental/sdc/vllm_scoring.py`: vLLM adapter specifications.
- `verl/trainer/ppo/ray_trainer.py`: end-to-end trainer lifecycle.
- `verl/workers/utils/losses.py`: actor-loss composition.
- `verl/workers/engine/fsdp/transformer_impl.py`: teacher engines, scoring,
  SFT, synchronization, and checkpoints.
- `verl/workers/engine_workers.py`: Ray worker RPCs.

## 1. End-to-end flow

```text
rollout generation
    -> verifier reward and explicit semantic outcome
    -> collect success/failure examples
    -> optional DAPO filtering/accumulation
    -> attach sdc_failure_mask to final actor batch
    -> prepare old log probabilities
    -> score failed rows under success/failure teachers
    -> actor forward for current log probabilities
    -> base policy loss + SDC loss
    -> actor update
    -> periodic teacher SFT refresh
    -> checkpoint and rollout-weight synchronization
```

## 2. Semantic outcome contract

SDC requires exactly one explicit binary outcome per response row:

```text
sdc_outcome in {0, 1}
```

Accepted fields are `batch.sdc_outcome`, `batch.acc`, or reward extra-info
fields named `sdc_outcome`, `acc`, `correct`, or `correctness`.

`validate_sdc_outcome()` in `data_collector.py:25` rejects non-binary, NaN,
Inf, and incorrectly shaped values. `extract_sdc_outcome()` at line 45 never
falls back to shaped token rewards. Therefore DAPO penalties, KL shaping, and
partial credit do not silently become SDC labels.

Reward managers expose the semantic value only when the verifier supplies it:

- `verl/workers/reward_manager/naive.py:93`
- `verl/workers/reward_manager/dapo.py:153`
- `verl/workers/reward_manager/batch.py:129`

## 3. Replay buffers

The trainer creates two collectors:

```python
SuccessDataCollector(target_reward=1)
FailureDataCollector(target_reward=0)
```

They are implemented by `OutcomeDataCollector` in
`verl/experimental/sdc/data_collector.py:77`.

Records contain prompt/response text fallbacks, compact `prompt_ids` and
`response_ids`, a binary reward, and metadata such as `uid` and
`global_policy_step`. The buffer supports bounded `recent` FIFO sampling or
`uniform` reservoir-style sampling.

Outcomes are collected before DAPO filtering at
`verl/trainer/ppo/ray_trainer.py:2901-2917`. A row can therefore be excluded
from the actor update while remaining available for teacher SFT.

## 4. Optional DAPO filtering

When `algorithm.filter_groups.enable=true`,
`_filter_dapo_groups()` at `ray_trainer.py:1011`:

1. Groups rows by `uid`.
2. Computes the configured group metric, usually `acc`.
3. Keeps only mixed groups containing at least one positive and one non-positive
   result.
4. Drops all-correct and all-wrong groups from the actor batch.

`_dapo_accumulate()` at line 1064 accumulates qualified groups across
generation batches until a full training batch exists. No actor update occurs
between those generations, preserving on-policy data.

SDC collection happens before filtering, but the actor failure mask is attached
only after the final filtered batch has been selected.

## 5. Final failure mask

After filtering/reordering, `_sdc_annotate_actor_batch()` at
`ray_trainer.py:2292` executes:

```python
outcome = validate_sdc_outcome(batch.batch["sdc_outcome"])
batch.batch["sdc_outcome"] = outcome
batch.batch["sdc_failure_mask"] = ~outcome
```

The mask is sequence-level, shape `[batch]`. The loss broadcasts it against
`response_mask`, so SDC-active positions are response tokens belonging to
failed sequences. Successful rows remain in the ordinary base policy loss.

## 6. Teacher initialization and SFT

`sdc_init()` at `transformer_impl.py:1025` captures one detached CPU snapshot:

```text
theta_0 = actor parameters at SDC initialization
theta_success = clone(theta_0)
theta_failure = clone(theta_0)
```

The two teachers have independent AdamW optimizers. Full fine-tuning makes all
clone parameters trainable; LoRA mode optimizes only adapter parameters.

Teacher refresh runs every `sft_update_interval_policy_steps`. Records are
paired by prompt/group UID using `pair_outcome_records_prompt_matched()` in
`sft_trainer.py:73`. Both branches receive equal counts and a common
response-token budget.

The response-only SFT objective is:

```text
L_SFT = mean_valid_examples(
            -sum(response-token log probabilities)
             / number_of_response_tokens
        )
```

Prompt tokens provide context but do not contribute to the loss. The engine
implementation is `_sdc_sft_model()` at `transformer_impl.py:1430`.

The refresh order is success SFT, failure SFT, equal-count/token-budget checks,
base mixing, model synchronization, and readiness-counter updates.

## 7. Frozen-base mixing

After every paired refresh:

```text
theta_success <- gamma * theta_success + (1 - gamma) * theta_0
theta_failure <- gamma * theta_failure + (1 - gamma) * theta_0
```

`gamma` defaults to `0.9`. Floating-point tensors use an FP32 temporary;
integer buffers are copied from the base. See `mix_state_with_base()` in
`verl/experimental/sdc/sdc_state.py:46`.

LoRA mode mixes only trainable adapter parameters. Base mixing changes model
parameters, not AdamW moments or step counters.

## 8. Teacher scoring

`_sdc_compute_model_log_probs()` at `ray_trainer.py:2550` returns:

```python
{
    "sdc_success_log_probs": ...,  # log pi_success
    "sdc_failure_log_probs": ...,  # log pi_failure
}
```

Only failed rows are scored; successful rows are zero-filled.

### HF backend

`sdc_compute_success_failure_log_probs()` at
`transformer_impl.py:1112` pads the batch, selects failure rows, moves both
sidecars to the actor device, scores bounded micro-batches, extracts shifted
response-token log probabilities, and offloads sidecars to CPU in
`auto_offload` mode.

### vLLM/LoRA backend

vLLM mode requires `model.lora_rank > 0` and
`reuse_rollout_log_probs=true`. The adapters are:

```text
success -> sdc_success, int_id=124
failure -> sdc_failure, int_id=125
```

They are defined in `vllm_scoring.py:8`, exported by
`sdc_export_vllm_adapters()` at `transformer_impl.py:1745`, and loaded through
`load_sdc_lora_adapters()` at `llm_server.py:472`.

vLLM receives existing prompt+response token IDs and returns prompt log
probabilities without generating a new response. The response span is extracted
by `score_sdc_logprobs()` at `vllm_async_server.py:611`.

## 9. Old and current actor log probabilities

The actor uses:

```text
pi_old   rollout or frozen pre-update policy
pi_theta current differentiable actor
```

Old values are selected in `ray_trainer.py:2955-3042`:

1. Reuse rollout-time `rollout_log_probs` when configured; this is required by
   vLLM SDC.
2. Otherwise run a separate pre-update forward through
   `_compute_old_log_prob()` at `ray_trainer.py:2025`.
3. With one PPO epoch and one logical mini-batch, use detached scores from the
   training forward, making the ratio exactly one and avoiding an extra forward.

Current log probabilities are produced during `_update_actor()` at
`ray_trainer.py:2062` and consumed as `model_output["log_probs"]` in
`losses.py:60`.

## 10. SDC actor-loss algebra

The core implementation is `compute_sdc_loss()` at
`verl/experimental/sdc/sdc_loss.py:57`. For active failed response tokens `A`:

```text
raw_log_ratio_t = log pi_theta_t - log pi_old_t
ratio_t         = exp(clamp(raw_log_ratio_t, -20, 20))
c_t             = log pi_failure_t - log pi_success_t
center          = mean_A(c_t)
rms             = sqrt(mean_A((c_t - center)^2))
norm_c_t        = (c_t - center) / max(rms, 1e-8)
```

The final loss is:

```text
L_SDC = beta * reliability_gate
        * sum_{t in A}(ratio_t * norm_c_t)
        / global_number_of_active_failed_response_tokens
```

The complete actor objective is:

```text
L_actor = L_base_policy + L_SDC
```

Teacher and old-policy log probabilities are detached; the actor gradient
enters through `ratio_t`. SDC always uses fixed `token-mean` aggregation and
requires importance weighting. If either teacher is not ready or `beta == 0`,
the SDC loss is exactly zero.

## 11. Distributed execution

Before micro-batching, the engine computes:

```text
global_active_token_count = all_reduce(
    sum(response_mask AND sdc_failure_mask)
)
```

This avoids an all-reduce per micro-batch. See `forward_backward_batch()` at
`transformer_impl.py:711`.

In distributed SFT, rank zero performs optimizer work; results are broadcast and
refreshed teacher weights are synchronized to other ranks. Worker RPCs are
registered in `engine_workers.py:744`.

Sidecar residency is `auto_offload` or `resident`. Resident mode requires an
explicit `sdc_release_residency()` call before waking vLLM or syncing rollout
weights.

## 12. Reliability gate

`_sdc_probe_discriminator_quality()` at `ray_trainer.py:2413` samples up to 64
successful rows, scores them under both teachers, and compares them with failed
row contrasts:

```text
AUC = P(discriminant_failed > discriminant_success)
      + 0.5 * P(tie)
ema <- 0.9 * ema + 0.1 * current_auc
reliability_gate = clamp(2 * (ema - 0.5), 0, 1)
```

The gate multiplies `L_SDC`. Probe failures retain the previous gate.

## 12b. SDC-TR zero-advantage repair (`sdc_tr_degenerate_coef`)

`compute_policy_loss_sdc_tr()` in `verl/trainer/ppo/core_algos.py` implements the
trust-region form of SDC: the teacher contrast tilts the PPO importance ratio
instead of being added as a separate loss term.

**The problem.** Group-centered advantages (Dr.GRPO / GRPO) subtract the group
mean. A group in which every sampled response is wrong therefore has `A == 0` on
every one of its tokens. The objective is `-A * tilted_ratio`, so the whole group
produces **no gradient at all** — the hardest prompts are silently dropped from
the update. The tilt cannot fix this, because it is a multiplicative factor on a
quantity that is already zero.

**The repair.** The opt-in knob
`actor_rollout_ref.actor.policy_loss.sdc_tr_degenerate_coef` (default `0.0`)
injects a pseudo-advantage on the *active failed* tokens whose advantage is
exactly zero:

```text
mu_eff            = sdc_tr_degenerate_coef * reliability_gate
c_t               = normalize(log pi_failure_t - log pi_success_t)   # detached
degenerate_mask   = response_mask AND failure_mask AND (A == 0)
pseudo_A_t        = -mu_eff * c_t          on degenerate_mask, else 0
A_eff_t           = A_t + pseudo_A_t
```

`c_t` is the same detached, centered-and-RMS-normalized teacher contrast the tilt
uses, so the two share one normalization and one reliability gate.

**Sign convention.** `c_t > 0` means the *failure* teacher assigns the token more
probability than the success teacher, i.e. the token looks like failure-specific
behaviour. Then `pseudo_A_t < 0`, and gradient descent **lowers** its log-prob.
`c_t < 0` means the *success* teacher wins, `pseudo_A_t > 0`, and descent
**raises** the log-prob. The pseudo-advantage is detached and is consumed by the
normal PPO clip / dual-clip path, so the trust region and `clip_ratio_c` still
bound the update exactly as for a real advantage.

**Safety properties.**
- `sdc_tr_degenerate_coef == 0.0` (the default) is bit-for-bit identical to the
  previous SDC-TR implementation; the repair branch never runs.
- The repair also requires both teachers to be available; without them `mu_eff`
  has no contrast to act on and the branch is skipped.
- `mu_eff` is scaled by the reliability gate (Section 12), so an untrusted
  discriminator injects (near) nothing.
- Only tokens with `A == 0` are touched. Groups with real learning signal are
  untouched.

**Metrics.** `sdc_tr/degenerate_coef` (the configured constant),
`sdc_tr/mu_eff` (after gating), `sdc_tr/degenerate_token_fraction` (share of
response tokens that received a pseudo-advantage) and
`sdc_tr/pseudo_advantage_abs_mean` (its mean magnitude on those tokens).

**When to use it.** Enable it when `sdc_tr/degenerate_token_fraction` is large —
typically early math-RL training, where many groups are all-wrong — and progress
on hard prompts has stalled. Keep it at `0.0` for a clean SDC-TR baseline.
Launchers: `SDC_TR_DEGENERATE_COEF` in `examples/sdc_grid/common.sh`, with
ready-made arms in `examples/sdc_grid/sdc_tr_zeroadv/`.

## 13. Checkpoint format

```text
global_step_X/
├── actor/
└── sdc/
    ├── success_sft/
    │   ├── pytorch_model.bin
    │   └── optimizer.pt
    ├── failure_sft/
    │   ├── pytorch_model.bin
    │   └── optimizer.pt
    └── sdc_state.pt
```

`_sdc_write_checkpoint()` at `transformer_impl.py:1348` saves teacher models,
optimizers, update counters, readiness, `base_mix_gamma`, the frozen base or
LoRA adapter-base state, and both replay buffers.

Full-parameter mode embeds the frozen base only on the first save in a process;
later saves use a marker referencing that base. Loading first reinitializes
teachers from the restored actor, then overlays checkpointed teacher and
optimizer state.

## 14. Practical checklist

1. Define an explicit verifier-level binary outcome.
2. Keep separate success and failure replay buffers.
3. Initialize both teachers from one actor snapshot.
4. Keep teacher optimizers independent.
5. Train teachers with response-only SFT.
6. Apply the final failure mask after filtering/reordering.
7. Detach teacher and old-policy log probabilities.
8. Use actor-dependent importance weighting.
9. Normalize by the global failed response-token count.
10. Gate the loss until both teachers are trained.
11. Checkpoint teachers, optimizers, base state, counters, and buffers.
12. Manage teacher GPU memory when sharing a device with vLLM.

## 15. Current limitations

The sidecars are ordinary cloned modules rather than FSDP-sharded modules. In
full-parameter mode, each actor rank can hold two complete teacher clones. HF
scoring adds two serial teacher forwards, and full-parameter base mixing and
synchronization can be model-sized.

See `SDC_FSDP_PERFORMANCE_AUDIT_AND_PLAN.md` for the performance audit and
`SDC_MODULARIZATION_REPORT.md` for the functional audit and validation status.


## Repaired native-sharded implementation

The default SDC teacher mode is `full_fsdp`. The actor engine creates two independent
teachers through the same HF model construction, remove-padding, Liger, attention,
mixed-precision, device-mesh, and FSDP wrap path as the actor. FSDP2 uses
`fully_shard`; FSDP1 remains available through the existing FSDP wrapper.

The actor-to-teacher copy is local-shard to local-shard. No rank summons a complete
actor, gathers a rank-zero model, or broadcasts complete teacher weights. Each teacher
has its own AdamW optimizer. The immutable SDC base is stored as a CPU copy of each
rank's local parameter shard, and post-SFT interpolation is performed in FP32 on that
local shard.

### Distributed actor data flow

~~~text
rollout outcomes
  -> DAPO/group filtering and reordering
  -> attach sdc_failure_mask to the final actor batch
  -> HF or rollout-reused teacher log probabilities
  -> compute global failed-token count, contrast sum, and contrast squared-sum
  -> one detached global contrast center/RMS
  -> dynamic microbatching
  -> additive actor loss
~~~

The failure mask is attached only after the final filtering/reordering operation.
Teacher scores and old-policy log probabilities are detached. Failed rows are scored;
successful rows remain in the actor batch with an inactive SDC mask.

Before actor microbatches are created, the engine computes the failed-token count and
the first two contrast moments over the complete logical batch. The three FP32
statistics are reduced over the actor data-parallel group once. Every microbatch
receives the same detached global count, center, and RMS. The SDC denominator is
therefore invariant to rank count, dynamic batching, and microbatch partitioning.

The loss keeps importance-ratio clipping, the global failed-token denominator, the
readiness gate, and additive composition with the base policy loss. A zero failed-token
batch returns the zero SDC term. Supported old-log-probability paths are:

1. rollout reuse for the vLLM/shared-LoRA path;
2. a separate pre-update actor forward when reuse is disabled;
3. fused ratio-equals-one when the PPO batch and optimizer schedule satisfy the
   existing fusion preconditions.

### Paired distributed SFT

Success and failure records are prompt-paired before token budgeting. The common
response-token budget is calculated globally and applied before rank partitioning.
Each logical batch is padded with masked dummy rows to a multiple of the actor data
parallel size. Every rank executes both the success and failure forward/backward and
optimizer-step schedule, including when its local slice contains dummies.

`sft_batch_size` is the global logical number of records per teacher update.
`sft_max_token_len_per_gpu` is a local capacity limit. The sequence-mean loss is
scaled by the data-parallel size before FSDP gradient averaging, giving the same
global gradient as a single-process reference. The success and failure teachers are
trained serially, then each is mixed with its own immutable base shard.

### Residency and scoring

The two sharded teachers can remain resident through scoring and SFT when
`sidecar_residency=resident`. The trainer must call
`sdc_release_residency()` before vLLM wake-up or actor weight synchronization.
`auto_offload` remains a compatibility mode. Scoring runs both teachers serially
and returns only failed-row scores in the payload; the full actor batch receives
zero-filled inactive positions.

### SDC checkpoint v2

An SDC v2 directory contains:

~~~text
sdc/
  manifest.pt
  _SUCCESS
  runtime.pt
  base_rank_<world>_rank_<rank>.pt
  success_sft/
    model_world_size_<world>_rank_<rank>.pt
    optim_world_size_<world>_rank_<rank>.pt
    extra_state_world_size_<world>_rank_<rank>.pt
    fsdp_config.json
  failure_sft/
    model_world_size_<world>_rank_<rank>.pt
    optim_world_size_<world>_rank_<rank>.pt
    extra_state_world_size_<world>_rank_<rank>.pt
    fsdp_config.json
~~~

The native FSDP checkpoint manager writes model, optimizer, and RNG/extra shards.
The manifest records the format version, world size, FSDP strategy, teacher paths,
base metadata, counters, readiness, and global step. The completion marker is
written only after all rank shards and runtime metadata are present.

Loading rejects an absent marker, missing rank shard, world-size mismatch, missing
immutable base, or incomplete teacher directory. Legacy v1 checkpoints are read only
during migration: rank zero reads the full legacy payload, validates optimizer-ID to
parameter-FQN coverage, broadcasts the migration payload, loads the base and teacher
states into native FSDP shards, restores both AdamW states, and immediately writes v2.
A slim legacy checkpoint fails clearly when its referenced base checkpoint is absent.

### Explicit shared-LoRA mode

`teacher_mode=shared_lora` requires a LoRA actor, vLLM scoring, and rollout
log-probability reuse. It creates one FSDP2 backbone with named `default`/success and
failure adapters, freezes the backbone, and gives each adapter its own optimizer.
Adapter activation is serialized for SFT and scoring. Base interpolation and vLLM
exports are adapter-only; updating one adapter does not update the backbone or the
other adapter.
