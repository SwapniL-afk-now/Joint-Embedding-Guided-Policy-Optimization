# SDC-GRPO entrypoints

This directory contains the ONLY scripts needed to run and evaluate
GRPO + SDC (Success-Failure Discriminative Contrast) training.

## Main file

### `run_sdc_grpo_fsdp.sh` — THE training launcher

Everything else in this repo is upstream verl examples, legacy JEPA/FEPO
experiments, or tooling. To train, run this one script:

```bash
bash examples/sdc_grpo/run_sdc_grpo_fsdp.sh
# override model/data via env or .env at the repo root:
MODEL_PATH=... DATA_PATH=... bash examples/sdc_grpo/run_sdc_grpo_fsdp.sh
```

What it does: pins GPUs 0,1; sources the repo `.env`; resolves the venv
python; sets CUDA/vLLM allocator options; builds the hydra command with all
`custom_sdc.*` settings; launches FSDP training with the HF scoring backend
and full-parameter sidecars by default.

All launcher knobs live as env variables near the top of the script
(model/data paths, batch sizes, SDC beta/intervals, timing). The canonical
algorithm needs no variant flags: prompt-matched teachers, centered and
scale-normalized contrast are always on. `beta` is the single algorithm
hyperparameter.

## Satellites

| File | Purpose |
|---|---|
| `run_repro_eval.sh` | Evaluate checkpoints after training |
| `run_sdc_sweep.sh` | Hyperparameter sweeps (rarely needed) |

## Notes

- Training writes `sdc/timing/*` metrics (scoring, success/failure SFT,
  base mixing, model sync, checkpoint save) — use them to locate remaining
  bottlenecks before optimizing anything.
- Health gates while training: `sdc/quality_auc` should stay well above 0.5
  (it certifies the contrast signal is informative), and
  `sdc/importance_weight_clip_fraction` should stay low.
- Everything under `examples/sdc*/presets` serves other base algorithms
  (DAPO, Dr.GRPO, NGRPO, AVSPO) and is not part of the default path.
