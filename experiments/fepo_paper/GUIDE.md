# Per-script guide

One entry per bash file: what it fills, what must exist first, how to run it, what to
check while it runs, and what to do when it finishes.

`RUNBOOK.md` has the program-level ordering and disk policy. This file is the detail.
Read the **Before any arm** section once, then jump to the arm you are launching.

Every arm follows the same shape:

```bash
cd /workspace/Joint-Embedding-Guided-Policy-Optimization
ARM_GPU=1 bash experiments/fepo_paper/<arm>.sh
```

Run it under tmux so it survives a dropped connection:

```bash
tmux new-session -d -s fepo_m06
tmux send-keys -t fepo_m06 'cd /workspace/Joint-Embedding-Guided-Policy-Optimization' Enter
tmux send-keys -t fepo_m06 'ARM_GPU=1 bash experiments/fepo_paper/m06_dapo_fepo.sh' Enter
```

---

## Before any arm

```bash
bash experiments/fepo_paper/00_setup.sh
```

That runs the self-checks, builds the probe set, downloads the 7B and Llama models,
and builds the cross-domain parquets. It is idempotent, so re-run it freely. All
code the program needs is implemented and tested — there is nothing left to write.

`_common.sh` refuses to launch on a missing model, missing data, an existing
checkpoint dir, under 30G free disk, or an unknown `custom_tafr_grpo.*` key. A
misconfigured arm therefore costs seconds, not 13 GPU-h. If an arm exits immediately,
read the message — it names the exact prerequisite.

**Never edit an arm script to change a shared setting.** Shared settings live in
`_common.sh`; changing one arm silently breaks every comparison in the paper.

---

## Group M — Table 1, the 1.5B plug-in matrix

The paper's core claim is base-objective independence: the *same* regularizer helps
GRPO, GSPO and DAPO. That claim lives or dies on these four pairs being matched, so
each pair differs in exactly one thing — FEPO on or off.

Baseline arms (`m01`, `m03`, `m05`, `m07`, `m08`) run with `diagnostic_only=true`:
TAFR stays enabled so the anchor and failure models are built and their KLs logged,
but they contribute exactly zero to the loss. This is what puts a KL number on rows
that have no regularizer, and it is why Tables 2 and 3 need no extra runs. Do **not**
"simplify" these to `TAFR_ENABLE=false` — that empties four table columns.

| Arm | Fills |
|---|---|
| `m01_grpo.sh` | Table 1 GRPO row |
| `m02_grpo_fepo.sh` | Table 1 FEPO[GRPO] row |
| `m03_gspo.sh` | Table 1 GSPO row |
| `m04_gspo_fepo.sh` | Table 1 FEPO[GSPO] row |
| `m05_dapo.sh` | Tables 1, 4 DAPO row |
| `m06_dapo_fepo.sh` | Tables 1, 4 DAPO+FEPO row |
| `m07_ngrpo.sh` | Table 1 NGRPO row |
| `m08_avspo.sh` | Table 1 AVSPO row |
| `m09_ngrpo_fepo.sh` | Table 1 FEPO[NGRPO] row |
| `m10_avspo_fepo.sh` | Table 1 FEPO[AVSPO] row |

### `m06_dapo_fepo.sh` — run this one first

It is the headline configuration: Table 1's FEPO[DAPO] row and Table 4's FEPO row.
(Table 3's "full FEPO" row and Table 8's sweep centre are now `m02` -- GRPO base,
not DAPO, to cut cost on the ancillary studies.) It is also the **canary**: two
gates must pass before you commit the other 21 arms.

After the first validation round (step 0, since `VAL_BEFORE_TRAIN=true`):

```bash
python experiments/fepo_paper/analyze.py check val_dumps/fepo-m06_dapo_fepo-<TAG>
```

Probe prompts must spread across all five difficulty buckets. If everything sits at
p=0 or p=1, sampled validation did not reach the config and **Table 2 is unbuildable** —
stop and fix before burning the rest of the budget.

Second gate, in wandb on any `diagnostic_only` arm: `tafr_grpo/kl_anchor` non-zero
**and** `tafr_grpo/loss_total` exactly 0.0. Both, or the flag is not taking effect.

### `m05_dapo.sh` — run this second

Feeds Tables 1 and 4. Once `m05` and `m06` are both done, Table 4 needs no more runs.

### `m01`–`m04`

Independent; run two at a time, one per GPU. `m03`/`m04` add
`loss_mode=gspo`; the launcher hardcodes `vanilla` but `"$@"` is appended last, so
hydra's last-wins gives GSPO. Confirm in the wandb config if you want to be sure.

### `m07`-`m10`

NGRPO and AVSPO are implemented as advantage estimators in
`verl/trainer/ppo/core_algos.py`. Both leave mixed groups byte-identical to GRPO and
differ only on homogeneous groups (asserted in `test_paper_additions.py`), so the
baseline comparison against the GRPO row is controlled.

`m07`/`m08` (baselines) run with `diagnostic_only=true`, which `validate_tafr_config`
permits under any estimator because the regularizer contributes nothing -- it is
what puts a KL number on a row with no FEPO term.

`m09`/`m10` (FEPO composed with NGRPO/AVSPO) do NOT set `diagnostic_only` -- FEPO
actually contributes. This is a real composition, not a diagnostic no-op: k3_legacy's
gate and KL terms come from `_tafr_group_reward_mean` (raw per-uid reward) and
log-probs only, never from the advantage tensor, so they compose validly with any
estimator. `m10` is the sharpest mechanism contrast in the paper -- AVSPO repairs the
collapsed scalar advantage, FEPO leaves it untouched and supplies a learned
direction -- so it tests whether the two are complementary or redundant.

---

## Group S — Table 1, 7B scaling

| Arm | Fills |
|---|---|
| `s01_dapo_7b.sh` | Table 1 7B DAPO row |
| `s02_dapo_fepo_7b.sh` | Table 1 7B FEPO[DAPO] row |

~45 GPU-h each — by far the most expensive arms, so run them only after the 1.5B
matrix has confirmed the effect is there to scale.

`00_setup.sh` downloads `Qwen2.5-Math-7B-Instruct`. Both arms drop
`ROLLOUT_GPU_MEM_UTIL` to 0.6 and `PPO_MINI_BATCH_SIZE` to 8 for memory headroom; if
they still OOM, lower `ROLLOUT_GPU_MEM_UTIL` before touching anything else.

**Do not re-tune β/γ/K for 7B.** Transfer of the 1.5B-selected hyperparameters without
retuning *is* the claim being tested. Retuning to rescue a weak result would answer a
different question than the one the table asks.

---

## Group A — Table 3, five-way ablation

Base objective is **GRPO, not DAPO** -- these are ancillary mechanism studies, and
DAPO is Table 1's own object of study, not needed as scaffolding here. Row 1 is
`m01` (plain GRPO) and row 5 is `m02` (GRPO+FEPO), both already run. Only three
new arms.

| Arm | Row | Tests |
|---|---|---|
| `a01_anchor_only.sh` | + anchor only | is the gain just EMA stabilization? |
| `a02_failure_only.sh` | + failure repulsion only | is unconstrained repulsion enough? |
| `a03_nogate.sh` | + both, no gate | is conditioning on failure rate necessary? |

`a02` is the one most likely to diverge: with the anchor pull removed there is nothing
opposing the replay push. That is a *result*, not a failure — record where it broke
rather than tuning it into stability, since instability without the anchor is exactly
what the ablation is meant to demonstrate.

`a03` uses `gate_mode=constant`, verified to change the loss (a no-op flag would
make this ablation a duplicate of `m06`).

---

## Group N — Table 7, negative-control

`n01_dapo_negce.sh`

Base objective is GRPO, not DAPO -- matches `m02`, not `m06`. Replaces the learned
failure density with uniform negative cross-entropy on the **same** failed replay
buffer. This is what stops the paper being dismissed as "negative training on failed
tokens", so the buffer must be identical to `m02`'s — same capacity, same sampling,
same refresh cadence. If the buffer differs, the control proves nothing.

Report Avg., Recovery@100 and the easy-bucket delta. The easy-bucket delta is the
informative one: uniform token suppression should damage easy prompts in a way a
learned failure density does not.

---

## Group P — Table 8, sensitivity sweep

Base objective is GRPO, not DAPO. One factor at a time around the `m02` centre
(β₀ = anchor 1.0 / replay 1.0, γ₀ = 0.9, K₀ = 5). The centre needs no run — it *is*
`m02`.

| Arm | Factor |
|---|---|
| `p01_beta_half.sh` | β = 0.5·β₀ (anchor 0.5 / replay 0.5) |
| `p02_beta_dbl.sh` | β = 2·β₀ (anchor 2.0 / replay 2.0) |
| `p03_gamma_half.sh` | γ 0.9 → 0.45 |
| `p04_gamma_dbl.sh` | γ 0.9 → 0.99 |
| `p05_k_half.sh` | refresh every 3 steps |
| `p06_k_dbl.sh` | refresh every 10 steps |

Both β components scale together on purpose. Halving only one changes the
anchor:replay *ratio*, which is a second factor and breaks "one factor at a time".

Two caption honesty notes: `p04` is 0.99, not literally 2·γ₀ (2×0.9 is not a valid
decay), and `p05` is 3 steps, not 2.5. Say so rather than writing "2γ₀" and "0.5K₀".

`p02` is the divergence risk — its replay weight (2.0) exceeds the absolute value that
made wandb run `oyul6igu` (anchor 0.5 / replay 1.0) blow up after step ~144, even
though `p02`'s 1:1 ratio is less replay-heavy than `oyul6igu`'s 1:2. Watch grad_norm
from the start.

---

## Group X — Table 5, cross-domain

`x01_llama_grpo.sh`, `x02_llama_grpo_fepo.sh`

The reward is rule-verifiable (`verl/utils/reward_score/instruction_following.py`),
scored as Hard Success Rate: 1.0 only if every constraint on the prompt holds. That
all-or-nothing shape is deliberate — it is what produces the homogeneous all-wrong
groups the method targets, and partial credit would smooth away the failure mode
under test.

`prepare_crossdomain_data.py` **drops** IFEval prompts carrying a constraint the
verifier cannot check exactly, rather than scoring them leniently. Report the retained
count as "IFEval (verifiable subset, n=...)". FollowBench is best-effort: its judged
categories are excluded on purpose, and if extraction yields nothing the script says
so and Table 5 is reported on IFEval alone.

These two arms are the ones most exposed to external breakage (a gated Llama repo, HF
dataset drift). Everything else in the program is independent of them.

MMLU is a regression check only. It is deliberately absent from `BEST_CKPT_SOURCES`;
never let it drive checkpoint selection.

---

## While an arm runs

Watch three things in wandb:

1. **`actor/grad_norm`** — the known divergence signature (wandb `oyul6igu`): flat and
   low for ~100 steps, then a creep (0.1 → ~3 by step 130), then runaway (→108 by
   step 153) as entropy hits 10 and accuracy collapses to 0.19. **Early calm proves
   nothing** — that run looked healthy through step 120. Kill an arm on *sustained
   creep past ~3*, not on a single spike.
2. **`dapo_filter/num_gen_batches`** (DAPO arms only) — how many generation batches
   dynamic sampling needed per update. Rising over the run is expected as groups go
   all-correct. If it pins at the budget, qualified groups have dried up and the
   arm is training on short batches; note it.
3. **`tafr_grpo/failure_sft_loss`** — if it plateaus high (~0.29 rather than ~0.07),
   the failure clone is not fitting, the replay term is inert, and the arm is paying
   the anchor cost for nothing. That is what happened at β 0.5/1.0 previously.

---

## When an arm finishes

```bash
# 1. final numbers: multi-seed, sampled -- NOT the single-seed in-training validation
python eval_math_multiseed_batched.py --model checkpoints/fepo-paper/<exp>/best_pass_at_1 ...

# 2. mechanism numbers (Table 2)
python experiments/fepo_paper/analyze.py recovery val_dumps/<exp>

# 3. systems numbers (Table 4, DAPO arms)
python experiments/fepo_paper/analyze.py systems <entity>/fepo-paper/<run_id> --target 0.60

# 4. reclaim disk -- required, not optional
rm -rf checkpoints/fepo-paper/<exp>
```

Keep `val_dumps/<exp>/` forever. It is small, and it is the only source for Table 2 —
deleting it means re-running the arm.

Step 4 is mandatory. A run that fills the disk stalls, and because the launcher
hardcodes `resume_mode=auto`, the retry silently *continues the stalled checkpoint*
instead of starting clean — so a disk problem turns into a quietly corrupted arm.

---

## What goes in the paper

- One training seed per config. Table 1's "Paired Δ" is a paired **eval-seed** delta —
  decoding variance, not training variance. State it plainly.
- Any per-benchmark difference smaller than the eval-seed std is not a result.
- Pre-declare the Table 4 target (default MATH-500 pass@1 = 0.60) before looking at
  curves, not after.
