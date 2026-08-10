# FEPO ICLR 2027 agentic-review draft - compact evaluation

Main file: `main.tex`

Current title: **Failure Is a Distribution: Failure-Escape Policy Optimization for Verifiable LLM Reinforcement Learning**

This revision keeps the GSPO-style paper structure while deliberately reducing the experimental burden. The empirical section now tests only the claims necessary for a strong review:

1. **Breadth at 1.5B:** GRPO, GSPO, and DAPO versus the corresponding FEPO plug-in, with NGRPO and AVSPO as targeted homogeneous-group baselines.
2. **Scale at 7B:** DAPO versus FEPO[DAPO] only.
3. **Mechanism:** deterministic Greedy Recovery@50/100 on AMC23 + MinervaMath + OlympiadBench using n=1, temperature 0.
4. **Ablation:** base, anchor-only, failure-only, ungated anchor+failure, and full FEPO.
5. **Practicality:** sec/update, peak memory, rollout tokens to target, and GPU-hours to target.
6. **Robustness:** a minimal beta / anchor-EMA / failure-refresh sweep on 1.5B.
7. **Domain transfer:** one small verifier-based instruction-following comparison, GRPO versus GRPO+FEPO.
8. **Simple negative control:** negative CE on the same failed replay, in the appendix.

The paper intentionally does **not** require full 7B baseline breadth, verifier-noise sweeps, group-size sweeps, failure-type taxonomies, buffer-capacity grids, MDP-GRPO composition, or AVSPO+FEPO composition for the central claim.

## Verified implementation correspondence
The anchor formulation is unchanged: `D_KL(pi_actor || pi_anchor)`, with the same forward-KL estimator, gradient derivation, EMA update, and drift treatment used before the replay correction. Only the failure/replay side is corrected to the code-aligned `D_KL(pi_replay || pi_actor)` (written as `D_KL(pi_failure || pi_actor)` in the theory).

## Compile

```bash
latexmk -pdf main.tex
```

The main scientific text remains within nine pages; references and appendices follow.

### Current experimental framing
GRPO is the primary baseline. Use GRPO for all mechanism, ablation, sensitivity, and systems studies. Table 1 uses identical method rows at 1.5B and 7B for matched scaling comparisons. GSPO and DAPO remain transfer checks rather than the baseline for auxiliary experiments.

Result-table organization: Table 1 is the matched math comparison at 1.5B/7B; Table 2 is the code-generation generalization table. Do not duplicate the same GRPO math results in a second table.
