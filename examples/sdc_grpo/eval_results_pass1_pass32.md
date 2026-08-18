# SDC vs GRPO — held-out eval, pass@1 and pass@32

Checkpoints: GRPO `grpo-1gpu-seed31415-20260730/best_pass_at_8` (**step 350**) vs
SDC `sdc-prod-b05/best` (**step 280**). Both merged to HF and served with vLLM.

Protocol: math500 + amc23 + aime24 + aime26 (600 problems), **n=32** samples,
seeds **42, 52, 62, 72, 82**, temperature 0.6, top_p 0.95, max_tokens 3072,
`max_num_seqs=2048`, `max_num_batched_tokens=131072`.
Run with `eval_math_multiseed_batched.py`; raw JSON in `/workspace/eval_ckpts/results_*.json`.

## Primary result — all 5 seeds, mean ± std

| metric | GRPO step350 | SDC step280 | Δ |
|---|---|---|---|
| **pass@1** | 35.15 ± 1.44 | 34.86 ± 1.96 | **-0.28** |
| **pass@32** | 58.90 ± 1.53 | 60.48 ± 1.02 | **+1.58** |

### Per benchmark, all 5 seeds

| benchmark | n_q | metric | GRPO | SDC | Δ |
|---|---|---|---|---|---|
| math500 | 500 | pass@1 | 65.08 ± 0.74 | 65.28 ± 0.90 | +0.20 |
| math500 | 500 | pass@32 | 81.44 ± 0.08 | 82.44 ± 0.48 | +1.00 |
| amc23 | 40 | pass@1 | 55.50 ± 6.96 | 53.50 ± 7.52 | -2.00 |
| amc23 | 40 | pass@32 | 89.50 ± 4.00 | 91.50 ± 2.55 | +2.00 |
| aime24 | 30 | pass@1 | 10.00 ± 2.11 | 11.33 ± 2.67 | +1.33 |
| aime24 | 30 | pass@32 | 35.33 ± 4.52 | 37.33 ± 2.49 | +2.00 |
| aime26 | 30 | pass@1 | 10.00 ± 2.11 | 9.33 ± 2.49 | -0.67 |
| aime26 | 30 | pass@32 | 29.33 ± 2.49 | 30.67 ± 4.42 | +1.33 |

### Per seed, MACRO Δ (SDC − GRPO)

| seed | 42 | 52 | 62 | 72 | 82 |
|---|---|---|---|---|---|
| pass@1 | +1.11 | +2.91 | -1.55 | -1.26 | -2.62 |
| pass@32 | +1.66 | +0.36 | -0.27 | +2.12 | +4.05 |

pass@1 swings +2.91 to −2.62 across seeds — a 5.5 pt range, ~10x the 5-seed effect.
pass@32 is positive in 4/5 seeds and in all ten 3-seed subsets.

## Seed-selected variants — NOT RESULTS, kept only to show selection sensitivity

Choosing the best 3 of 5 seeds *per metric*. Each row is a different sample, so these
do not describe one experiment and must not be reported as such.

| metric | best 3 seeds | GRPO | SDC | Δ selected | Δ all 5 |
|---|---|---|---|---|---|
| pass@1 | 42, 52, 72 | 35.09 ± 1.43 | 36.01 ± 1.60 | +0.92 | -0.28 |
| pass@32 | 42, 72, 82 | 58.07 ± 1.47 | 60.68 ± 1.26 | +2.61 | +1.58 |

Selection flips pass@1 from −0.28 to +0.92 and inflates pass@32 by 65%.
The sign of pass@32 is never in question under any subset; pass@1's is entirely selection-dependent.

## Defensible claim

> SDC trades no measurable top-1 accuracy (pass@1 −0.28) for **+1.58 pts pass@32** (4/5 seeds),
> consistent with its preserved rollout diversity (training collapse rate 6% vs GRPO's 21%).

Caveats: one seed per training arm; step 350 vs 280 is not length-matched; both are
*selected* checkpoints but GRPO's was chosen over 5 validation seeds and SDC's over 1;
aime24/aime26 have 30 problems each, so one problem = 3.3 pts.
