# FEPO paper experiment runbook

22 training runs, ~395 GPU-h, ~8-9 days on 2 GPUs. Fills all 8 tables and both
figures of `preview/main.tex`.

Run one arm at a time:

```bash
cd /workspace/Joint-Embedding-Guided-Policy-Optimization
ARM_GPU=1 bash experiments/fepo_paper/m06_dapo_fepo.sh
```

Each arm script is ~5 lines: it sources `_common.sh` (all shared config) and sets
only its own deltas. **Change shared settings in `_common.sh`, never in an arm** --
changing one arm silently breaks every comparison the paper makes.

---

## Why only 22 runs

Five of the eight tables need **zero dedicated runs**. They are post-hoc reads of
metrics the Table 1 runs already emit -- but only if the instrumentation is
configured before those runs start -- `_common.sh` already does that. Skip the
probe set or sampled validation and you need ~20 more runs to recover the same numbers.

| Table | Runs | Source |
|---|---|---|
| 1 main results | 12 | `m01`-`m10`, `s01`-`s02` |
| 2 mechanism (Recovery@Δ, buckets) | **0** | probe dumps → `analyze.py recovery` |
| 3 ablation | 3 | `a01`-`a03` (GRPO base, not DAPO -- + `m02` reused as full-FEPO row) |
| 4 systems accounting | **0** | wandb → `analyze.py systems` (DAPO row: `m05`/`m06`) |
| 5 cross-domain | 2 | `x01`-`x02` |
| 6 paired plug-in (appx) | **0** | derived from Table 1 |
| 7 negative control (appx) | 1 | `n01` (GRPO base, not DAPO) |
| 8 sensitivity (appx) | 6 | `p01`-`p06` (GRPO base, centre = `m02`) |
| Figs 1-2 | **0** | Table 1 runs |

---

## Step 0 — setup (once)

```bash
bash experiments/fepo_paper/00_setup.sh
```

Runs the self-checks, builds the probe set, downloads `Qwen2.5-Math-7B-Instruct` and
`Llama-3.2-3B-Instruct`, and builds the cross-domain parquets. Idempotent.

`_common.sh` fails fast on any missing asset, so a bad launch costs seconds, not hours.

## Step 1 — nothing to implement

All code the program needs is in place and covered by two suites:

```bash
python experiments/fepo_paper/test_paper_additions.py   # 16 passed
python experiments/fepo_paper/test_dapo_accumulate.py   #  6 passed
```

**Still owed to the tex (paper edit, not code):** the setup paragraph's β/γ/K,
temperature and response cap do not match what actually runs. `_common.sh` holds the
real values; copy them across.

The KL directions now match the paper on both terms. The anchor estimator was fixed
(`anchor_kl_direction=forward`); `legacy` reproduces the old behaviour if you need to
A/B against runs made before the fix.

## Step 2 — canary

```bash
ARM_GPU=1 bash experiments/fepo_paper/m06_dapo_fepo.sh
```

This one arm is the headline config and the Table 4 FEPO row. (Table 3's "full FEPO"
row and the Table 8 sweep centre now come from `m02` -- GRPO base, not DAPO -- see
below.) After its first validation round, **both gates must pass before launching
anything else**:

```bash
python experiments/fepo_paper/analyze.py check val_dumps/fepo-m06_dapo_fepo-<TAG>
```
Must show probe prompts spread across all five difficulty buckets. If everything sits
at p=0 or p=1, sampled validation did not take effect and Table 2 is unbuildable.

Second gate, in wandb: `tafr_grpo/kl_anchor` non-zero **and** `tafr_grpo/loss_total`
exactly 0.0 on a `diagnostic_only` arm. Both conditions, or the flag is not taking effect.

## Step 3 — the rest, in this order

1. `m05_dapo` — the reference row for Tables 1 and 4.
2. `m01`-`m04` — two at a time, one per GPU. `m02` (GRPO+FEPO) additionally feeds
   Table 3's full-FEPO row and the Table 8 sweep centre.
3. `a01`-`a03`, `n01`, `p01`-`p06` — all 1.5B, GRPO-based (not DAPO), cheap, fully parallel.
4. `m07`-`m10` — NGRPO / AVSPO baselines and their FEPO compositions.
5. `s01`, `s02` — 7B, ~45 GPU-h each.
6. `x01`, `x02` — last. These depend on a gated model repo and HF datasets, so they
   are the most likely to break for external reasons; nothing else depends on them.

## Step 4 — per-run harvest, then delete

Final table numbers come from **offline multi-seed eval**, not from in-training
validation (which is single-seed and exists only for curves and checkpoint selection):

```bash
python eval_math_multiseed_batched.py --model checkpoints/fepo-paper/<exp>/best_pass_at_1 ...
python experiments/fepo_paper/analyze.py recovery val_dumps/<exp>
rm -rf checkpoints/fepo-paper/<exp>     # after the eval JSON is saved
```

---

## Disk hygiene — hard requirement

~117G free; a run at `save_freq=50` produced 47G. `_common.sh` sets `save_freq=100`
and the launcher hardcodes `max_actor_ckpt_to_keep=1`, holding a run near 7G.

**Never leave more than 3 runs' checkpoints resident.** `launch()` refuses to start
below 30G free. This matters more than it sounds: a run that fills the disk stalls,
and because the launcher hardcodes `resume_mode=auto`, the retry silently continues
the stalled checkpoint instead of starting clean.

`val_dumps/` is small and must be **kept** — it is the only source for Table 2.

## Health check on every run

The known divergence signature, from wandb run `oyul6igu`: grad_norm and entropy sit
low and flat for ~100 steps, then grad_norm creeps (0.1 → ~3 by step 130), then runs
away (→108 by step 153) as entropy hits 10 and accuracy collapses to 0.19.

Early calm proves nothing — that run looked healthy through step 120. **Watch for
sustained grad_norm creep past ~3**, and kill the arm if you see it. `a02_failure_only`
(replay push with no anchor pull) and `p02_beta_dbl` are the most likely to diverge.

## Honesty constraints for the tex

- One training seed per config. The Table 1 "Paired Δ" column is a paired **eval-seed**
  delta — decoding variance, not training variance. State this explicitly; a reviewer
  will notice if it is blurred.
- Any per-benchmark difference smaller than the eval-seed std is not a result.
- DAPO dynamic sampling now genuinely resamples (`_dapo_accumulate`), so the DAPO
  arms hold a full `train_batch_size` of qualified groups. Report
  `dapo_filter/num_gen_batches` — it is the extra generation cost DAPO pays.
- FEPO buffers failures and computes the gate on the RAW group, before DAPO filtering.
  Say so: it is what lets the method see the all-wrong groups DAPO discards.
- Pre-declare the Table 4 target (default: MATH-500 pass@1 = 0.60) **now**, not after
  seeing the curves.
