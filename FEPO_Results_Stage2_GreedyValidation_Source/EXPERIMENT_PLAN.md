# FEPO minimum convincing experiment plan — GRPO-primary revision

The goal is not to maximize experiment count. GRPO is the primary baseline and is used for all costly mechanism, ablation, sensitivity, and systems studies. GSPO and DAPO are retained only where they test transfer of FEPO across distinct group-relative RL algorithms.

## 1. Main matched comparison — Qwen2.5-Math-1.5B-Instruct and 7B-Instruct
Use the same method rows at both scales, with matched prompt pools, decoding settings, rollout-token budgets, and checkpoint rules:

- GRPO
- GRPO + FEPO
- GSPO
- GSPO + FEPO
- DAPO
- DAPO + FEPO
- NGRPO
- AVSPO

Current completed greedy pass@1 evaluation covers MATH-500, AMC23, AIME24, AIME25, OlympiadBench, MinervaMath, HumanEval+, MBPP+, and LiveCodeBench. Evaluation uses one decoded response per problem at temperature 0, reported over five training seeds (3407, 31415, 27182, 16180, 42).

Table 1 is the matched mathematical-reasoning comparison and contains identical method rows at 1.5B and 7B. The completed 1.5B GRPO / GRPO + FEPO math results are populated there; the remaining transfer and 7B rows stay as placeholders. Table 2 reports code-generation generalization separately on HumanEval+, MBPP+, and LiveCodeBench. GRPO / GRPO + FEPO is the principal comparison; GSPO and DAPO are transfer checks rather than the baseline used for auxiliary experiments.


## Completed result block
- Greedy pass@1, temperature 0, n=1 decoding: completed for GRPO and GRPO + FEPO on nine benchmarks.
- Five training seeds: 3407, 31415, 27182, 16180, 42.
- Remaining transfer/scaling, recovery, ablation, sensitivity, and systems results are still pending.

## 2. Mechanism metrics — reuse the same checkpoints
No extra training runs are required for:

- Recovery@50
- Recovery@100
- gain by initial success-rate bucket: `p=0`, `(0,.25]`, `(.25,.5]`, `(.5,.75]`, `(.75,1]`
- actor-to-anchor KL
- replay-to-actor KL

Use the GRPO runs for the difficulty-bucket analysis. The critical signature is that FEPO gains concentrate at low initial success rate.

## 3. Five-way ablation — 1.5B GRPO only

- GRPO
- GRPO + anchor only
- GRPO + failure repulsion only
- GRPO + anchor + failure with constant/no gate
- GRPO + full FEPO

Do not repeat this ablation on DAPO or GSPO. Their role in the paper is transfer, not mechanism isolation.

## 4. Simple negative control — 1.5B GRPO only
Use exactly the same failed replay buffer:

- GRPO
- GRPO + negative CE on failed trajectories
- GRPO + FEPO

Report Avg., Greedy Recovery@100, and deterministic validation delta. This protects the paper from being reduced to “negative training on failed tokens.”

## 5. Systems accounting — from GRPO paired runs
Report:

- sec/update
- peak GPU memory
- rollout tokens to one pre-declared target
- GPU-hours to the same target
- final average

Do not repeat systems profiling on DAPO. The GRPO pair is the canonical cost comparison and avoids paying for a second expensive systems study.

## 6. Minimal robustness sweep — 1.5B GRPO only
One factor at a time:

- beta: `0.5 beta0`, `beta0`, `2 beta0`
- anchor EMA gamma: `0.5 gamma0`, `gamma0`, `2 gamma0`
- failure-policy refresh cadence: `0.5 K0`, `K0`, `2 K0`

Keep buffer capacity fixed. No large hyperparameter grid is required.

## 7. One compact non-math check
Llama-3.2-3B-Instruct:

- GRPO
- GRPO + FEPO

Use IFEval / FollowBench rule-verifiable constraints. This is only a domain-transfer sanity check, not a second full benchmark suite.

## Removed from the required plan
Unless later reviews specifically demand them, do not spend compute on:

- DAPO-based ablation, negative-control, sensitivity, or systems tables
- XRPO and M-GRPO training runs
- AVSPO + FEPO composition
- MDP-GRPO composition
- verifier-noise sweeps
- continuous-reward experiments
- group-size sweeps
- failure-type labeling/taxonomy
- buffer-capacity sweep
- large estimator-clipping sweeps

## DAPO ordering for the transfer runs
1. Generate and score the raw group.
2. Compute `g` on the raw group.
3. Buffer failed trajectories and record actor/anchor/failure log-probabilities.
4. Apply DAPO dynamic sampling / homogeneous-group filtering.
5. Construct the DAPO surrogate.
6. Add FEPO and update the actor.

## Verified implementation correspondence
Checked against the implemented replay-loss path: the failure term is `D_KL(pi_replay || pi_actor)`, not `D_KL(pi_actor || pi_failure)`. The anchor side remains `D_KL(pi_actor || pi_anchor)`, with the same estimator and EMA treatment. Future audits should modify only the failure/replay side unless the anchor code itself changes.

## Baseline stopping rule
Do not add TPO, DGPO, CARE, or EMA-PG merely because an agentic reviewer names them. TPO, DGPO, and CARE are positioned explicitly in Related Work because they target different mechanisms. EMA-PG overlaps with the anchor component, which is already isolated by the GRPO anchor-only ablation. If a future real reviewer requires one additional baseline, EMA-PG is the highest-priority optional run; otherwise the current controls are sufficient to test FEPO's stated mechanism.

## Shared rollout/reference configuration used by the review draft
- group size: `G=8`
- sampling temperature: `0.6`
- top-p: `0.95`
- response cap: `2048` tokens
- beta: `0.02`
- anchor EMA gamma: `0.02`
- failure buffer: FIFO, 4096 responses
- failure refresh: every 20 actor steps
- failure-policy refresh minibatch: 128 failed responses, sampled uniformly
- raw failure gate: no cap and no EMA smoothing

Replace these with the exact implementation settings if the real training configuration differs.
