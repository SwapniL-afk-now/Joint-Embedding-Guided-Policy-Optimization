# Compact DeepSeek-Math-7B experiment plan

This is the reduced experiment program for `deepseek-ai/deepseek-math-7b-instruct`.
It replaces the paper's 1.5B auxiliary runs and does not require Table 1/3 to be
rerun through this plan. The implementation uses the repository's existing
TAFR/FEPO code path (the paper's anchor + learned failed-trajectory reference).

## What is run

| Arm | Purpose |
|---|---|
| `d01_grpo_7b.sh` | GRPO control; reference/anchor diagnostics enabled but loss disabled |
| `d02_grpo_fepo_7b.sh` | Full reference method |
| `d03_anchor_only_7b.sh` | Anchor-only ablation |
| `d04_replay_only_7b.sh` | Replay/reference-only ablation |
| `d05_nogate_7b.sh` | Full method with constant rather than failure-rate gate |
| `d06_negce_7b.sh` | Negative-cross-entropy control on the same failed replay |
| `d07_recovery_grpo_7b.sh`, `d08_recovery_fepo_7b.sh` | Sampled fixed-probe Recovery@50/@100 pair |

The main pair is two configurations. The three ablation variants and the
negative-CE control use GRPO only; DAPO/NGRPO/AVSPO are intentionally omitted to
avoid multiplying the expensive 7B matrix.

## Prerequisites

```bash
cd /workspace/Joint-Embedding-Guided-Policy-Optimization
bash experiments/fepo_paper/deepseek7b_setup.sh
```

The compact setup checks/tests the implementation, builds the recovery probe, and resolves or downloads the DeepSeek
snapshot. The arm resolver accepts `DEEPSEEK_MODEL_PATH`; otherwise it searches
`/workspace/models` and the Hugging Face cache.

DeepSeek-Math-7B has a 4096-token context window. The default compact settings
keep prompt plus response within that limit and use two GPUs with LoRA. On a
larger machine, override `DRGRPO_USE_LORA=false` only after a pilot confirms
memory headroom.

## Dry run and launch

Always inspect the expansion first:

```bash
bash experiments/fepo_paper/run_deepseek7b_compact.sh --dry-run
```

The launcher is confirmation-gated. It defaults to one seed, which is suitable
for a smoke/canary run:

```bash
bash experiments/fepo_paper/run_deepseek7b_compact.sh --confirm --phase main
```

For the final study, use five seeds for the main pair and three for auxiliary
variants (the runs are sequential and expensive):

```bash
MAIN_SEEDS=3407,31415,27182,16180,42 \
AUX_SEEDS=3407,31415,27182 GPUS=0,1 \
bash experiments/fepo_paper/run_deepseek7b_compact.sh --confirm --phase all
```

Use `--phase ablations` to run only the four auxiliary arms. Every seed receives
a unique `TAG`, preventing checkpoint and W&B collisions. The launcher's existing
free-space and existing-checkpoint guards remain active.

## Recovery protocol

Recovery requires two additional runs (`--phase recovery`). The main arms use greedy n=1 validation, which cannot produce per-prompt success rates. The recovery arms use the fixed `probe256.parquet`, n=8 sampled rollouts at temperature 0.6, and validation before training. Run:

```bash
AUX_SEEDS=3407 bash experiments/fepo_paper/run_deepseek7b_compact.sh --confirm --phase recovery
```

Then check the dump and compare the paired arms:

```bash
python experiments/fepo_paper/analyze.py check val_dumps/<control-dump>
python experiments/fepo_paper/analyze.py compare val_dumps/<control-dump> val_dumps/<fepo-dump> --deltas 50 100
```

## Post-hoc measurements (no additional training)

From `d07` and `d08` sampled recovery validation dumps, and from `d01`/`d02` main checkpoints for learning/cost diagnostics, compute:

1. Deterministic Recovery@50 and Recovery@100 on AMC23, MinervaMath, and
   OlympiadBench. At checkpoint `t0`, select prompts that are greedily incorrect;
   re-evaluate the same prompts after 50/100 actor steps.
2. Accuracy versus rollout-token budget and recovery curves.
3. Runtime/update, peak memory, GPU-hours to a predeclared target, and final
   average accuracy.
4. Gate statistics, all-zero/all-success group rates, actor-anchor KL,
   replay-actor KL, actor-old KL, replay age, and clipping/clamping rates.

Preserve `val_dumps/` and save evaluation JSON before deleting checkpoints.

## Reporting controls

- Keep verifier, data, group size, rollout-token budget, decoding, and checkpoint
  selection identical across all arms.
- Report the fraction of homogeneous groups; a math-specialized model may solve
  too many prompts or fail too many for the gate to be informative.
- Do not retune the ablations after observing instability. Replay-only divergence
  is itself evidence for the anchor's stabilizing role.
- Report training-seed uncertainty separately from decoding/evaluation uncertainty.
- Table 6 (IFEval/FollowBench) is omitted: DeepSeek-Math-7B is math-specialized,
  so it is not a clean replacement for the paper's instruction-following model.
