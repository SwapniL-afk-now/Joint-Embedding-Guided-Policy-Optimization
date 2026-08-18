# TAFR vs GRPO — why training accuracy and eval accuracy disagree

**Runs.** TAFR `5j6rgx1l` (`tafr-prod-b05`, β=0.5, stopped at step 405) vs GRPO
`fw7zf51z` (`grpo-1gpu-seed31415-20260730`, 666/666, capped at step 404 here).
Qwen2.5-Math-1.5B, amcaime32k, 1 GPU, lr 1e-6, batch 64, rollout n=8 — identical
train and eval parquets. Eval = mean over amc23/aime24/aime25/aime26, 8 samples,
~130 problems.

---

## 1 & 2. GRPO's training accuracy climbs; its training pass@8 does not

Steps 1–50 → 355–404:

| | GRPO | TAFR |
|---|---|---|
| train accuracy (mean of 8) | **39.6 → 51.4%** | 39.2 → 41.4% |
| train pass@8 | 76.8 → 79.2% | 77.0 → 77.3% |

GRPO gains 11.8 points of accuracy and 2.4 points of pass@8. It is **not solving
problems it previously could not solve** — it is raising the probability of answers
already present in its 8 samples. pass@8 is blind to that; pass@1 is sensitive to it.

Your point 2 is the load-bearing one: **both policies have the same reachable
solution set and differ only in how sharply they commit to it.**

## 3. The sharpening is a measurable collapse

| diagnostic (steps 1–50 → 355–404) | GRPO | TAFR |
|---|---|---|
| unique answer ratio @ k=8 | **0.495 → 0.398** | 0.510 → 0.503 |
| exploration collapse rate | **6.2 → 20.7%** | 6.8 → 6.0% |
| all-correct group rate | 5.7 → 18.0% | — |
| mixed (learnable) group rate | 71.2 → 61.2% | — |
| answer entropy @ k | 1.090 → 0.820 | — |
| gradient norm | 0.091 → 0.074 | 0.142 → 0.130 |
| train acc − val pass@1 (gap) | 18.2 → 25.5 pts | 17.5 → 19.9 pts |

GRPO's collapse rate **triples**: one in five training prompts produces eight copies
of the same answer. A collapsed group has zero advantage variance and therefore
contributes **zero gradient**. Its mixed-group rate falls 10 points and its gradient
norm decays accordingly. TAFR's collapse rate is flat-to-falling and its gradient
norm stays **~1.75× higher** for the whole run.

That is the direct answer to your point 3: TAFR's training accuracy is lower *and*
its held-out accuracy is higher, so its generalization gap is 5.6 points narrower
and widening more slowly. Nearly all of GRPO's extra training accuracy is confined
to the training distribution.

## 4. What drives TAFR — and the evidence against it

**Mechanism.** The β=0.5 failure-anchored regularizer injects a gradient component
worth ~11% of GRPO's advantage RMS that is *not* aligned with "commit harder to the
answer you already picked." That keeps rollouts diverse → keeps advantage variance
alive → keeps the effective gradient ~75% larger. GRPO spends its budget sharpening
onto the training set; TAFR spends the same budget staying spread out.

Supporting metrics, all stable across the run:
- `tafr_adv/regularization_to_grpo_rms_ratio` 0.112 → 0.117 (β target is being hit;
  the term is neither vanishing nor dominating)
- `tafr_adv/active_group_fraction` ≈ 0.94
- `tafr_grpo/failure_sft_loss` flat at 0.774 (failure model tracks a moving policy
  rather than converging and going stale)

**The counter-evidence — the advantage is decaying, not growing.** Paired per-eval
difference in 4-set pass@1:

| phase | GRPO | TAFR | Δ |
|---|---|---|---|
| steps 20–110 | 20.68% | 21.92% | **+1.24 pts** |
| steps 120–210 | 21.45% | 23.75% | **+2.30 pts** |
| steps 220–310 | 20.84% | 22.00% | +1.16 pts |
| steps 320–400 | 21.47% | 21.76% | **+0.29 pts** |

TAFR is **already ahead at the first eval** (step 20: 21.3% vs 19.4%), peaks in
steps 120–210, then decays to +0.3 by 320–400. GRPO is closing, not falling behind.
The honest framing: a mild early advantage that GRPO erodes as it converges — not a
compounding gain. Whether the curves cross after 404 is unmeasured; that's where the
run stopped.

## 5. What to tune, priority order

Each knob is here because a logged metric says it is mis-set or untested.

1. **`TAFR_BETA` 0.5 → 1.0, 2.0** — highest value. β is a target ratio of
   regularizer to GRPO advantage RMS and it is being hit precisely (0.112, flat).
   The effect is real but small and nothing indicates a stability limit: entropy
   flat, no clip-fraction blowup, no gradient spikes. 200 steps each; read collapse
   rate and val pass@1.
2. **`TAFR_SFT_UPDATE_INTERVAL` 5 → 2 or 10** — the failure model refreshes every 5
   GRPO steps and its SFT loss never descends, so it is permanently chasing the
   policy. 2 tests whether a sharper failure model gives a stronger signal; 10 tests
   whether the signal is mostly noise. Either direction is free information.
3. **`TAFR_EMA_GAMMA` (0.9) / `TAFR_MIX_ETA` (0.5)** — how much of the anchor is the
   moving EMA vs the fixed reference. Since the advantage *decays* as the policy
   moves away from the reference, this pair is the prime suspect for the decay. Try
   η = 0.8 and see whether the 320–400 gap stops closing.
4. **`TAFR_FAILURE_DATA_MAX_SIZE` (unlimited) / `_SAMPLING` (recent)** — the buffer
   is unbounded, so failures the policy has since fixed still train the failure
   model. Cap it (~20k) so it describes *current* failure modes.
5. **`rollout.n` 8 → 16** — not a TAFR knob, but it targets the actual bottleneck:
   with n=8 and GRPO at 20% collapse, a fifth of every batch carries zero gradient.
   n=16 restores advantage variance for both arms, and is the clean control for
   "is TAFR just compensating for too-small groups?"
6. **`actor.optim.lr` 1e-6** — lowest priority. GRPO's collapse is the signature of
   too much sharpening pressure. If TAFR's benefit vanishes at lr 5e-7, TAFR is an
   lr correction wearing a mechanism — worth knowing before publishing it as one.

## What this does not establish

- **One seed per arm.** The measured GRPO seed effect in this project is |Δ| 0.70
  pts on 4-set pass@1; TAFR's full-window mean advantage is +1.28 pts — larger, but
  the same order. A second TAFR seed is required to separate effect from seed.
- **The headline depends on the eval set.** Over all four sets TAFR wins 29/39
  paired steps at +1.28 pts. Drop aime25 and it wins 38/39 at +2.39 pts. aime25 is
  the one set where TAFR is behind. Report the 4-set number as primary unless there
  is a pre-registered reason to exclude it.
- **TAFR resumed from a step-10 checkpoint** with the failure model already at 69
  updates, and was ahead at the first eval. Part of the gap may be initialization.
- **Unequal footing.** TAFR stopped at 405/666; all comparisons cap GRPO at 404.
- The mechanism claim in §4 is *consistent with* the collapse/gradient/diversity
  metrics, not causally isolated. The β sweep and the n=16 control are what would
  isolate it.
