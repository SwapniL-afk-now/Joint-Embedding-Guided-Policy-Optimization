# Multi-GPU gradient deficit: investigation report

**Date:** 2026-07-30
**Affected runs:** `advproj-b0.05-2gpu-20260730` (killed at step 382/666), and in principle
any multi-GPU run in this fork, including plain GRPO.
**Status:** cause characterized and worked around; exact defect **not yet located in code**.

---

## 1. Summary

A 2-GPU run of the advantage-projection arm underperformed its 1-GPU GRPO baseline on both
training and validation metrics. The apparent cause was assumed to be the new auxiliary
loss. It was not.

**With the auxiliary loss completely disabled (`representation_beta=0`), moving the same
config and the same data from 1 GPU to 2 GPUs collapses `actor/grad_norm` by ~3.3x.** The
2-GPU run was therefore training at a fraction of its intended effective learning rate. The
representation objective was never implicated.

Two independent problems were found. Both are documented here because the first one is what
delayed finding the second.

---

## 2. Problem 1 — comparison against the wrong baseline

### What happened

The advantage-projection arm trains on
`/workspace/jepa-grpo-cache/data/amcaime32k/train.parquet` (`math_dapo`, 21,366 rows).
It was being compared against `logs_baseline_nojepa_20260725.log` (wandb `2pk6qu21`), which
trains on `deepscaler_preview_train.parquet` (`deepscaler-preview-dataset`, 40,309 rows) —
a **different dataset**, with a different system prompt and a different user suffix.

The correct baseline is:

| | value |
|---|---|
| wandb run | `u84okom0` |
| checkpoint dir | `checkpoints/grpo-qwen-3b/grpo-amcaime32k-mathl35-20260728` |
| console log | `wandb/run-20260728_223257-u84okom0/files/output.log` |
| training data | `amcaime32k/train.parquet` (same as advproj) |
| steps | 666 |
| **world_size** | **1 (single GPU)** — from `global_step_666/actor/fsdp_config.json` |

### How the mistake showed up

Against the wrong (deepscaler) baseline, `train/pass_at_8` differed by **+12.2 points in
steps 1-50**. Since both runs start from the same checkpoint, a gap that large before
learning can differentiate cannot be an effect of the loss. That anomaly is what triggered
the configuration audit.

### Corrected comparison (38 step-matched evals, steps 10-380)

| metric | advproj (2 GPU, β=0.05) | GRPO (1 GPU) | Δ | advproj wins |
|---|---|---|---|---|
| val pass@1 | 0.2114 | **0.2147** | **-0.0032** | 14/38 |
| val pass@8 | **0.3746** | 0.3729 | +0.0017 | 21/38 |

Training accuracy, step-matched:

| steps | advproj | GRPO | Δ |
|---|---|---|---|
| 1-50 | 0.3947 | 0.4044 | -0.010 |
| 121-200 | 0.4396 | 0.4666 | -0.027 |
| 281-379 | 0.4622 | 0.5076 | **-0.045** |

So the arm was behind, and falling further behind — which is what prompted the gradient
investigation below.

### Rule adopted

Before any cross-run comparison, verify that both runs match on:
1. `data.train_files`
2. `fsdp_config.json` -> `world_size`
3. step-1 `train/accuracy` (should be near-identical from a shared init)

**Any metric gap present at step 1 is a configuration difference, never a learning effect.**

---

## 3. Problem 2 — the multi-GPU gradient deficit

### How it was found

After correcting the baseline, `actor/grad_norm` was compared at **step 1**, chosen
deliberately: at step 1 no optimizer step has yet altered the weights, so both runs hold the
same model and (verified) near-identical data.

| | step 1 | step 2 | step 3 |
|---|---|---|---|
| GRPO baseline (1 GPU) | 0.0996 | 0.1028 | 0.0939 |
| advproj (2 GPU, β=0.05) | **0.0136** | 0.0292 | 0.0236 |
| `train/accuracy` (advproj vs GRPO, step 1) | 0.3633 vs 0.3672 | | |

A 7.3x gradient gap with matched data and matched weights is a defect in the gradient path,
not a training outcome.

### Ruling out configuration

The two launchers were diffed. Every quantity affecting gradient scale is **identical**:

`optim.lr=1e-6`, `loss_agg_mode=token-mean`, `clip_ratio_low/high=0.2`, `clip_ratio_c=3.0`,
`use_kl_loss=false`, `kl_loss_coef=0.0`, `entropy_coeff=0`, `adv_estimator=grpo`,
`norm_adv_by_std_in_grpo=False`, `train_batch_size=64`, `ppo_mini_batch_size=64`,
`rollout.n=8`, `max_prompt_length=1024`, `max_response_length=3072`, `temperature=1.0`.

`verl/trainer/ppo/core_algos.py` was byte-identical between the two checkouts
(`/workspace/exploration` and this repo), including `agg_loss`.

### The isolating experiment

Three 4-step runs, identical data and config, varying only GPU count and β:

```
TOTAL_TRAINING_STEPS=4 SAVE_FREQ=1000 TEST_FREQ=1000 VAL_BEFORE_TRAIN=false \
NDEVICES_PER_NODE=<1|2> REP_BETA=<0.0|0.05> \
bash examples/adv_projection_trainer/run_qwen25_math_1_5b_fsdp.sh
```

| run | GPUs | β | step-1 `grad_norm` | interpretation |
|---|---|---|---|---|
| A | 1 | 0.00 | **0.09332** | reproduces GRPO (0.0996) — shared code path is clean |
| C | 1 | 0.05 | **0.1268** | auxiliary loss *adds* ~36% gradient energy |
| B | **2** | **0.00** | **0.02845** | **3.3x deficit with the aux loss OFF** |
| (killed run) | 2 | 0.05 | 0.0136 | both conditions combined |

Full series — A: 0.09332, 0.08919, 0.08244, 0.09015. B: 0.02845, 0.01701, 0.02269.

**Run B is decisive.** `representation_beta=0` disables the auxiliary loss entirely and
exercises only the stock GRPO path, yet the gradient is still ~3.3x short. The defect is in
the fork's data-parallel path and predates this work.

### Consequence for training dynamics

A 2-GPU run trains at a fraction of its intended effective learning rate. The symptoms are
deceptive because several of them superficially look like *better* behaviour:

| metric (steps 281-379) | advproj (2 GPU) | GRPO (1 GPU) | naive reading | actual cause |
|---|---|---|---|---|
| `actor/grad_norm` | 0.0189 | 0.0760 | — | 4x smaller updates |
| `train/accuracy` | 0.4622 | 0.5076 | worse | under-training |
| `exploration_collapse_rate` | 0.1427 | 0.1828 | "healthier" | under-training |
| `unique_answer_ratio_at_k` | 0.4548 | 0.3965 | "more diverse" | under-training |

The lower collapse rate and higher answer diversity were initially misread as the auxiliary
objective preserving exploration. They are simply the consequence of taking smaller steps.

---

## 4. What was tried

| # | Attempt | Outcome |
|---|---|---|
| 1 | Compare full-run and step-matched metrics vs `2pk6qu21` | Invalid — wrong dataset. Discarded. |
| 2 | Locate the true baseline via wandb run id, checkpoint dirs, `fsdp_config.json` | Found `u84okom0`, world_size 1 |
| 3 | Check train/eval contamination (exact prompt-set intersection, both datasets vs all 4 eval sets) | **0 overlap** in every pair — clean |
| 4 | Diff both launchers on all gradient-affecting hyperparameters | Identical — configuration ruled out |
| 5 | Diff `core_algos.py` / `agg_loss` between the two checkouts | Byte-identical — code drift ruled out |
| 6 | Read `agg_loss` token-mean normalization (`masked_sum / batch_num_tokens * dp_size`) | Reads correct; `batch_num_tokens` is all-reduced globally in `forward_backward_batch` |
| 7 | Read `optimizer_step` grad-norm measurement | Correct — `FSDP.clip_grad_norm_` reduces across shards, so the number is not a per-shard artifact |
| 8 | Inspect `accumulate_minibatch_grads` gating (forced on when β>0) | No-op in this config (`ppo_mini_batch_size == train_batch_size`, single mini-batch); also disabled at β=0, where the deficit still appears |
| 9 | Consider `pg_loss` being ~2x lower on 2 GPUs | Explained as a *metric reporting* artifact — metrics aggregate as SUM per rank and only rank 0 is logged. Does **not** explain `grad_norm`. |
| 10 | Three-run isolation experiment (A/B/C above) | **Located the cause: the DP path, independent of β** |

### Hypotheses considered and their status

| Hypothesis | Status |
|---|---|
| Auxiliary loss suppresses/cancels the policy gradient | **Refuted** — deficit persists at β=0 (run B); at 1 GPU the aux loss *increases* grad_norm (run C) |
| Missing `* dp_size` in loss normalization | Not supported by reading — the factor is present and applied. Would also predict exactly 2x, not 3.3x |
| `grad_norm` is a per-shard measurement | **Refuted** — FSDP's `clip_grad_norm_` is shard-aware |
| `accumulate_minibatch_grads` regression | **Refuted** — inactive at β=0, where the deficit still appears |
| Different training data / hyperparameters | **Refuted** — verified identical |

### Not yet explained

The measured ratio is **~3.3x, not 2x**. A single missing data-parallel averaging factor
would predict exactly `dp_size` = 2. The extra factor is unaccounted for. Candidate areas
not yet examined: micro-batch partitioning under `use_dynamic_bsz` with
`same_micro_num_in_dp=True`, and the FSDP gradient reduction/pre-divide configuration.

---

## 5. Workaround in force

Run all RL arms on **1 GPU**, matching the GRPO baseline's `world_size: 1`. This is required
for comparability regardless of the bug — a 2-GPU arm must never be compared against a
1-GPU baseline.

Relaunched run:

```
CUDA_VISIBLE_DEVICES=0 VERL_CONSOLE_FULL_METRICS=1 \
NDEVICES_PER_NODE=1 REP_BETA=0.05 TOTAL_TRAINING_STEPS=666 \
SAVE_FREQ=10 TEST_FREQ=10 VAL_BEFORE_TRAIN=false \
PROJECT_NAME=grpo-qwen-3b EXPERIMENT_NAME=advproj-b0.05-1gpu-20260730 \
bash examples/adv_projection_trainer/run_qwen25_math_1_5b_fsdp.sh
```

Step-1 verification of the relaunch:

| check | value | expected |
|---|---|---|
| `actor/grad_norm` | 0.1183 | ~0.10-0.13 (GRPO 0.0996, run C 0.1268) — was 0.0136 |
| `train/accuracy` | 0.3945 | ~0.36-0.40 |
| `optimizer_step_calls_this_batch` | 1.0 | exactly 1 |
| `skipped_batch` | 0.0 | 0 |
| `representation/loss` | 0.4601 | aux active |

Cost: ~59 s/step on one GPU (~10h45m for 666 steps) versus ~34 s/step on two.

---

## 6. Incidental issue — disk exhaustion

During the diagnostics `/workspace` reached **100% (1.7 MB free)**. This surfaced as a
crash at checkpoint save:

```
RuntimeError: [enforce fail at inline_container.cc:858] .
  PytorchStreamWriter failed writing file data/13: file write failed
RuntimeError: [enforce fail at inline_container.cc:664] . unexpected pos 2990159168 vs 2990159060
```

It is a disk error, not a model or CUDA error, and it killed diagnostic run B at step 4
(its gradient measurements from steps 1-3 remained valid). Each run's checkpoint directory
is ~30-60 GB. 78 GB of throwaway diagnostic checkpoints were deleted; all experiment
checkpoints were preserved.

---

## 7. Reproduction recipe

To re-check the deficit on any future change:

1. Run 4 steps at `NDEVICES_PER_NODE=1` and 4 steps at `NDEVICES_PER_NODE=2`, both with
   `REP_BETA=0.0`, `SAVE_FREQ=1000 TEST_FREQ=1000 VAL_BEFORE_TRAIN=false`.
2. Compare **step-1** `actor/grad_norm`. Step 1 is before any weight update, so the two runs
   hold identical weights; any gap is structural.
3. Sanity-check that step-1 `train/accuracy` matches between the runs, confirming the data
   is the same.
4. Expected once fixed: the two `grad_norm` values agree within rollout sampling noise
   (~10%).

---

## 8. Files and identifiers

| item | path |
|---|---|
| correct GRPO baseline log | `wandb/run-20260728_223257-u84okom0/files/output.log` |
| correct GRPO baseline checkpoints | `checkpoints/grpo-qwen-3b/grpo-amcaime32k-mathl35-20260728` |
| wrong baseline (deepscaler, do not use) | `logs_baseline_nojepa_20260725.log` (wandb `2pk6qu21`) |
| killed 2-GPU run log | `logs_advproj.log` |
| corrected 1-GPU run log | `logs_advproj_1gpu.log` |
| loss normalization | `verl/trainer/ppo/core_algos.py::agg_loss` (~line 1138) |
| global batch stats / micro-batching | `verl/workers/engine/fsdp/transformer_impl.py::forward_backward_batch` (~line 621) |
| grad clip and optimizer step | `verl/workers/engine/fsdp/transformer_impl.py::optimizer_step` (~line 1259) |
| mini-batch accumulation gating | `verl/workers/engine_workers.py` (~line 347), `verl/workers/engine/base.py::train_batch` (~line 120) |
