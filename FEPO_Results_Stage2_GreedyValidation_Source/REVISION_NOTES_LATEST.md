# Latest revision: reverse replay KL with anchor side restored

- The verified failure/replay term remains `D_KL(pi_replay || pi_actor)`.
- Reverted all unnecessary anchor-side rewrites: the anchor remains the original forward `D_KL(pi_actor || pi_anchor)`, with the prior forward-KL gradient derivation, IW-k3 estimator/corollary, EMA update, and anchor-drift treatment restored.
- Only theory that mathematically depends on the failure KL orientation remains changed: the forward-forward contrastive-target reduction and global convexity claims stay removed, and the reverse replay term is analyzed as replay-weighted anti-imitation with proximal boundedness.
- The main gradient analysis now separates the unchanged forward anchor derivation from the corrected reverse failure derivation.
- The implementation appendix likewise restores the original forward-anchor estimator theorem verbatim in substance, then derives the reverse replay estimator separately.
- All empirical tables and figures remain placeholders.

## GRPO-primary experimental revision
- GRPO is now the canonical baseline for mechanism ablations, negative controls, sensitivity analysis, and systems accounting.
- DAPO-specific auxiliary tables were replaced with GRPO equivalents to reduce experimental cost while preserving DAPO as a transfer check in the main comparison.
- Table 1 now uses matched training rows for Qwen2.5-Math-1.5B-Instruct and Qwen2.5-Math-7B-Instruct: GRPO, GRPO+FEPO, GSPO, GSPO+FEPO, DAPO, DAPO+FEPO, NGRPO, and AVSPO.
- The success-rate bucket analysis is now defined on GRPO.
- The appendix negative-CE control is now GRPO-based.

- Removed the duplicated all-benchmark GRPO table. Table 1 is now the matched 1.5B/7B math comparison; Table 2 is a distinct code-generation generalization table (HumanEval+, MBPP+, LiveCodeBench).
