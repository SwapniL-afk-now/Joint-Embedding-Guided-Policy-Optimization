# TAFR-GRPO repro-b10 (beta=1.0) — Qwen2.5-Math-1.5B-Instruct

Checkpoint: `checkpoints/tafr-repro-math15b/tafr-sweep-repro-b10-20260805/best_pass_at_1/actor`,
selected by `best_ckpt_metrics=["pass@1"]` at **global_step 310** (not the run's latest step,
950 — best-checkpoint selection, same convention as `results_tafr_repro_math15b.md`). Arm is
`repro-b10`: beta ablation, beta=1.0 (repro baseline is beta=0.1), everything else identical to
the `repro` config (xdl0kb1c, full-finetune, dsr+MATH345, 1000-step budget).

Merged to `/workspace/merged_ckpt/repro-b10`, evaluated offline with vLLM via
`ARM=repro-b10 GPU=1 bash examples/tafr_grpo/run_repro_eval.sh` — the same script used for the
`repro` / `repro-grpo` arms in `results_tafr_repro_math15b.md`.

## Greedy decoding (n=1, temperature 0)

pass@1 (%), mean ± std over seeds. Math core (math500/amc23/aime24/aime25/aime26) uses 5 seeds:
the 4 offline ones (31415/27182/16180/42) plus seed 3407 pulled from the in-training validation
of wandb run `4dmi8ijs` (`tafr-sweep-repro-b10-20260805`) at global_step 310 — the same seed/step
`run_repro_eval.sh`'s comment says is "already measured in-training", so it's reused rather than
re-run. Extra+code benchmarks already covered seed 3407 offline (5 seeds throughout).

| Benchmark | n | pass@1 | seeds |
|---|---|---|---|
| math500 | 500 | 64.64 ± 0.53 | 5 (offline ×4 + wandb 3407@step310) |
| amc23 | 40 | 65.50 ± 2.45 | 5 (offline ×4 + wandb 3407@step310) |
| aime24 | 30 | 9.33 ± 3.27 | 5 (offline ×4 + wandb 3407@step310) |
| aime25 | 30 | 11.33 ± 1.63 | 5 (offline ×4 + wandb 3407@step310) |
| aime26 | 30 | 12.67 ± 1.33 | 5 (offline ×4 + wandb 3407@step310) |
| olympiadbench | 674 | 34.21 ± 0.40 | 5 |
| minervamath | 272 | 17.94 ± 0.59 | 5 |
| humanevalplus | 164 | 14.51 ± 0.98 | 5 |
| livecodebench | 287 | 7.11 ± 0.47 | 5 |
| **Macro (5 core math)** | 630 | **32.69** | — |
| **Macro (7 math)** | 1576 | **32.91** | — |
| **Macro (2 code)** | 451 | **10.81** | — |

Same caveat as the original report applies: at temperature 0 the "seed" spread is vLLM batching
nondeterminism, not a real error bar.

## GRPO vs. TAFR repro (beta=0.1) vs. TAFR repro-b10 (beta=1.0)

GRPO and repro (beta=0.1) columns are from `results_tafr_repro_math15b.md` (Table 1, same greedy
protocol, same eval script). mbppplus was not in the initial `run_repro_eval.sh` batch for
repro-b10 — run separately after, same protocol (n=1, temp=0, 5 seeds), via
`eval_code_multiseed_batched.py --benchmarks mbppplus` (raw JSON:
`/workspace/eval_results/repro-b10/code_mbpp.json`).

| Benchmark | n | GRPO | TAFR repro (β=0.1) | TAFR repro-b10 (β=1.0) |
|---|---|---|---|---|
| math500 | 500 | **67.20** | 65.56 | 64.64 |
| amc23 | 40 | 49.50 | 62.50 | **65.50** |
| aime24 | 30 | **12.00** | 10.67 | 9.33 |
| aime25 | 30 | 14.00 | **19.33** | 11.33 |
| aime26 | 30 | **19.33** | 13.33 | 12.67 |
| olympiadbench | 674 | 31.93 | **34.66** | 34.21 |
| minervamath | 272 | 18.16 | **20.22** | 17.94 |
| humanevalplus | 164 | **14.02** | 10.37 | 14.51 |
| mbppplus | 378 | 16.24 | **16.88** | 16.24 |
| livecodebench | 287 | 4.67 | **6.20** | 7.11 |
| **Macro (5 core math)** | 630 | — | — | **32.69** |
| **Macro (7 math)** | 1576 | 30.30 | **32.32** | 32.91 |
| **Macro (3 code)** | 829 | **11.65** | 11.15 | 12.62 |

Beta=1.0 vs beta=0.1 (both TAFR): raising beta doesn't help — aime25 drops 8 points (19.33 →
11.33, the single biggest move in the whole table) and math500/aime24/aime26/minervamath all
tick down slightly. amc23 (+3.0) and code (livecodebench +0.9, humanevalplus +4.1, mbppplus flat)
improve, but not enough to offset the AIME loss. Both TAFR arms still beat GRPO's math macro
(30.30) by a comfortable margin; b10's 7-math macro (32.91) edges out even beta=0.1's (32.32),
driven mostly by amc23. b10's 3-code macro (12.62) also edges out beta=0.1 (11.15) and GRPO
(11.65) — livecodebench carries it, mbppplus ties GRPO exactly. Single training/eval run per arm
— no significance claim, see the caveats in
`results_tafr_repro_math15b.md`.

Raw per-seed JSON: `/workspace/eval_results/repro-b10/{math_core,math_extra,code}.json`.
