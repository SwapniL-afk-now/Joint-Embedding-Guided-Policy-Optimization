# SDC / SDC-TR comparison grid

A 3-base x 3-mode grid (plus an alpha sweep for SDC-TR) that isolates the effect of
the SDC contrast term from the choice of base RL objective, using hyperparameters
inherited from the GXPO reference run so the SDC term is the only thing that varies.

Reference run: tmux `qwen25-1p5b-throughput0902` →
`/workspace/final-gxpo/Code/SFPO/experiments/gxpo_efficiency/qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh`.

## Layout

```
examples/sdc_grid/
  common.sh                    # single source of truth for every hyperparameter
  base/{drgrpo,dapo,avspo}.sh                      # base objective, no SDC term
  sdc/{drgrpo,dapo,avspo}.sh                        # base objective + additive SDC loss
  sdc_tr/{drgrpo,dapo,avspo}_a{1.0,0.9,0.7,0.5}.sh  # base objective + SDC folded into the PPO ratio
  queue.sh                     # sequential tmux runner
```

Every entrypoint is a 4-line wrapper (`BASE_ALGO=... SDC_MODE=... exec bash common.sh`);
`common.sh` is the only file that defines a value. This mirrors the existing
`examples/sdc/presets/*.sh` pattern in this repo.

## Grid

| | `base` (no SDC) | `sdc` (additive) | `sdc_tr` (trust-region) |
|---|---|---|---|
| **drgrpo** | `base/drgrpo.sh` | `sdc/drgrpo.sh` | `sdc_tr/drgrpo_a{1.0,0.9,0.7,0.5}.sh` |
| **dapo** | `base/dapo.sh` | `sdc/dapo.sh` | `sdc_tr/dapo_a{1.0,0.9,0.7,0.5}.sh` |
| **avspo** | `base/avspo.sh` | `sdc/avspo.sh` | `sdc_tr/avspo_a{1.0,0.9,0.7,0.5}.sh` |

18 scripts total. `ngrpo` is intentionally excluded from this grid — it requires
`policy_loss.loss_mode=ngrpo`, which `sdc_tr` would overwrite. Use
`examples/sdc/presets/ngrpo.sh` (the additive-SDC launcher) for NGRPO work.

`sdc_tr_alpha` controls how strongly the teacher contrast tilts the PPO importance
ratio: `1.0` is exact vanilla-PPO equivalence (bit-for-bit — see
`tests/trainer/test_sdc_tr_paper_fidelity.py`), lower values tilt harder. The `a1.0`
cell in each base row is therefore also the on-GPU sanity check: its loss curve
should track the matching `base/*.sh` run step-for-step.

## Running

```bash
# dry run: preflight + resolved config, no launch
bash examples/sdc_grid/base/drgrpo.sh --dry-run

# single arm, in its own tmux session (required by this repo's CLAUDE.md)
tmux new-session -d -s sdc-grid-drgrpo -c examples/sdc_grid
tmux send-keys -t sdc-grid-drgrpo "bash sdc_tr/drgrpo_a0.9.sh" C-m

# or let queue.sh manage the session and run several arms back-to-back
bash examples/sdc_grid/queue.sh sdc-grid-1 base/drgrpo sdc/drgrpo sdc_tr/drgrpo_a0.9
```

Any variable in `common.sh` can be overridden from the environment without editing a
file, e.g. `SDC_BETA=1.0 bash examples/sdc_grid/sdc/dapo.sh` or
`PPO_MAX_TOKEN_LEN_PER_GPU=131072 bash examples/sdc_grid/base/avspo.sh` for higher
throughput at the cost of exact token-budget parity with GXPO (numerically neutral —
loss aggregation is `token-mean` over pre-micro-batch global statistics, so it is
partition-invariant).

**GPU note:** at the time this grid was built, all 4 GPUs were saturated by the live
GXPO run. `queue.sh` refuses to start while GPUs already have >4 GB in use unless
passed `--force`; `common.sh` itself only warns. Wait for the GXPO run to finish, or
free GPUs, before launching for real.

Suggested first run once GPUs are free: `sdc_tr/drgrpo_a1.0.sh`. It is the cheapest
correctness check of the whole harness — its loss curve must match `base/drgrpo.sh`.

## What's identical to GXPO, and what isn't

`common.sh` inherits every learning-relevant GXPO setting verbatim: batch/minibatch
size, sequence lengths, seed, optimizer, clip ratios (per base), grad clip, actor KL
loss (`use_kl_loss=True, kl_loss_coef=0.01, low_var_kl` — matching GXPO, not this
repo's off-by-default `run_sdc_grpo_fsdp.sh`), FSDP size, attention backend, sampling
params, and trainer cadence. See the "INHERITED FROM GXPO" block in `common.sh` for
the full list with values.

A handful of settings **cannot** be identical because this repo's verl is a different
version from the GXPO fork (async server-mode rollout vs. sync colocated, several
renamed/relocated config keys). Every one of these is enumerated in the
`SDC_GRID_DIVERGENCES` block in `common.sh`, printed in every run's config header,
and checked for completeness by
`tests/trainer/test_sdc_grid_launchers_on_cpu.py::test_divergences_are_exhaustive` —
so a silent, undocumented drift from GXPO's config becomes a test failure rather than
a confound nobody notices six runs later.

This is also why `base/*.sh` **re-runs** the three baselines inside this repo instead
of reusing the live GXPO numbers: the engines aren't comparable enough for a
head-to-head, even though their configs are semantically identical.

## SDC hyperparameters (constant across every SDC/SDC-TR arm)

`beta=0.5`, `scoring_backend=hf` (full-parameter teachers), `base_mix_gamma=0.9`,
`min_sft_updates_before_use=2`, `sft_update_interval_policy_steps=5`,
`sft_lr=1e-5`, `sft_batch_size=8`, `importance_weight_clip=5.0`,
`sidecar_residency=auto_offload`. See `common.sh`'s "SDC hyperparameters" block for
the complete list.

## Tests

`tests/trainer/test_sdc_grid_launchers_on_cpu.py` — CPU-only, no GPU or model
weights required:
- every one of the 18 entrypoints exits 0 under `--dry-run`
- the GXPO-inherited settings actually match GXPO's launcher, key by key
- the divergence list is exhaustive (no undocumented drift either direction)
- each arm's resolved algorithm/loss/SDC config matches what its name claims
- the `a1.0` SDC-TR cells differ from their `sdc` sibling only in `loss_mode` /
  `sdc_tr_alpha`

Run: `PYTHONNOUSERSITE=1 <venv>/bin/python -m pytest tests/trainer/test_sdc_grid_launchers_on_cpu.py -v`
