# TAFR-GRPO vs GRPO — Qwen2.5-Math-1.5B-Instruct

**Setup.** Train: `ypwang61/One-Shot-RLVR-Datasets` dsr_sub + `EleutherAI/hendrycks_math` L3–5
(6601 problems). Prompt/response 1024/3072, actor lr 1e-6, GRPO n=8, `norm_adv_by_std_in_grpo=False`.
TAFR arm: `k3_legacy` loss, variant `full`, beta 0.1, EMA gamma 0.9, mix eta 0.5, SFT every 5 GRPO
steps at lr 5e-7, failure batch 64. Control is identical with `custom_tafr_grpo.enable=false`.
Train seed 3407. wandb project `tafr-repro-math15b` — TAFR `8sjh2hez`, GRPO `dxv253gl`.
TAFR stopped at step 574/1000; GRPO completed 1000/1000.

Checkpoints selected by `best_ckpt_metrics=["pass@1"]` on the 5-benchmark math macro at
val seed 3407: **TAFR step 550**, **GRPO step 410**. Both merged to HF format and evaluated
offline with vLLM.

---

## Table 1 — Greedy decoding (n=1, temperature 0)

pass@1 (%), mean ± std over 5 seeds (3407, 31415, 27182, 16180, 42).

| Benchmark | n | TAFR-GRPO | GRPO | Δ |
|---|---|---|---|---|
| math500 | 500 | 65.56 ± 0.26 | **67.20 ± 0.45** | −1.64 |
| amc23 | 40 | **62.50 ± 1.77** | 49.50 ± 1.12 | +13.00 |
| aime24 | 30 | 10.67 ± 1.49 | **12.00 ± 1.83** | −1.33 |
| aime25 | 30 | **19.33 ± 1.49** | 14.00 ± 2.79 | +5.33 |
| aime26 | 30 | 13.33 ± 2.36 | **19.33 ± 2.79** | −6.00 |
| olympiadbench | 674 | **34.66 ± 0.52** | 31.93 ± 0.44 | +2.73 |
| minervamath | 272 | **20.22 ± 0.64** | 18.16 ± 0.42 | +2.06 |
| humanevalplus | 164 | 10.37 ± 0.61 | **14.02 ± 0.75** | −3.66 |
| mbppplus | 378 | **16.88 ± 0.68** | 16.24 ± 0.71 | +0.63 |
| livecodebench | 287 | **6.20 ± 0.52** | 4.67 ± 0.47 | +1.53 |
| **Macro (10)** | 2405 | **25.97 ± 0.51** | 24.71 ± 0.27 | **+1.27** (5/5 seeds) |
| **Macro (7 math)** | 1576 | **32.32 ± 0.80** | 30.30 ± 0.39 | **+2.02** (5/5 seeds) |
| **Macro (3 code)** | 829 | 11.15 ± 0.25 | **11.65 ± 0.13** | **−0.50** (1/5 seeds) |

**The ± values here are not error bars.** At temperature 0 the seed does not change the sampled
tokens; the spread is vLLM batching nondeterminism. Verified directly: re-running the AIME sets at
10 seeds returned *identical* scores at all 10 (aime24 TAFR 10.00 ± 0.00). Do not run significance
tests on this table.

---

## Table 2 — Stochastic sampling (n=16, temperature 0.6, top-p 0.95)

avg@16 (%), mean ± std over 4 seeds (3407, 31415, 27182, 16180) = 64 samples per problem.
Currently AIME only; the remaining seven benchmarks are still to be run.

| Benchmark | n | TAFR-GRPO | GRPO | Δ | seeds won |
|---|---|---|---|---|---|
| aime24 | 30 | **11.20 ± 0.46** | 10.16 ± 0.80 | **+1.04** | 4/4 |
| aime25 | 30 | **11.20 ± 0.87** | 10.68 ± 1.23 | +0.52 | 2/4 |
| aime26 | 30 | **11.15 ± 0.52** | 10.05 ± 0.86 | **+1.09** | 4/4 |
| **Macro (3 AIME)** | 90 | **11.18 ± 0.40** | 10.30 ± 0.23 | **+0.89** | 4/4 |

Coverage, same runs — pass@16 (%):

| Benchmark | TAFR-GRPO | GRPO | Δ | seeds won |
|---|---|---|---|---|
| aime24 | 26.67 ± 4.71 | **32.50 ± 1.67** | −5.83 | 0/4 |
| aime25 | 30.00 ± 7.20 | 30.00 ± 2.72 | 0.00 | 1/4 |
| aime26 | 24.17 ± 1.67 | **25.00 ± 3.33** | −0.83 | 1/4 |
| **Macro (3 AIME)** | 26.94 ± 3.56 | **29.17 ± 1.90** | **−2.22** | 1/4 |

---

## Reading

- **Math: TAFR ahead by ~2 points macro**, on all 5 seeds. The most trustworthy single result is
  olympiadbench + minervamath — 946 problems, never used for checkpoint selection, TAFR ahead by
  +2.73 and +2.06 with no seed overlap between the arms.
- **Greedy pass@1 on 30-problem sets is unreliable.** Table 1 shows aime24 −1.33 and aime26 −6.00;
  both flip to TAFR wins under sampling (Table 2). One problem is 3.3 points, decided by a single
  deterministic rollout. Report AIME on avg@16, not greedy.
- **amc23's +13.00 is inflated.** Over full training the gap averages ~+6 points (TAFR mean 0.575
  vs GRPO 0.515 across all validation points). The rest is checkpoint-selection luck — TAFR's
  step 550 sits near its amc23 peak, GRPO's step 410 in a dip — plus a harness shift (GRPO scored
  0.575 in-training at step 410 vs 0.495 re-measured offline). Not a parsing or truncation failure:
  answer parse rate is 97.5–100% for both throughout.
- **GRPO narrows onto the training distribution.** Its math500 rose (0.638 → 0.680) while amc23 fell
  (~0.58 → ~0.50), and it degrades overall past step ~570 (macro 0.3069 early → 0.2936 over its last
  10 validation points). TAFR improved both.
- **Code is roughly neutral, not a loss.** The 3-benchmark macro is −0.50, but the direction is
  inconsistent: TAFR wins livecodebench (+1.53) and mbppplus (+0.63), loses humanevalplus (−3.66),
  which is the smallest of the three at 164 problems (6 problems). Neither model is competent at
  code (6–17% pass@1); both were trained purely on math with a math system prompt.
- **Trade-off: accuracy up, coverage down.** TAFR raises avg@16 but lowers pass@16 (−2.22 macro on
  AIME), i.e. it is more accurate per sample but solves fewer *distinct* problems given 16 tries.
  Consistent with TAFR's training-time `exploration_collapse_rate` of 0.59. A claim of "better
  single-shot accuracy" holds; "better reasoning coverage" does not.

## Caveats

- **Single training seed.** One TAFR run against one GRPO run. Validation seeds are held out;
  the training seed is not.
- **Unequal training length.** GRPO's best was selected from 99 validation points over 1000 steps,
  TAFR's from 57 over 574.
- **livecodebench here is the stdin/stdout subset** (287 problems) handled by this repo's
  `stdio_code` scorer, not the full LCB release — not comparable to published LCB numbers.

## Reproducing

```bash
.venv/bin/python -m verl.model_merger merge --backend fsdp \
  --local_dir checkpoints/tafr-repro-math15b/tafr-sweep-repro-20260804/best_pass_at_1/actor \
  --target_dir /workspace/merged_ckpt/repro
ARM=repro GPU=1 bash examples/tafr_grpo/run_repro_eval.sh   # ARM=repro-grpo GPU=0 for the control
```

Raw per-seed JSON in `/workspace/eval_results/{repro,repro-grpo}/`:
`math_core.json` (4 seeds), `math_core_3407.json`, `math_extra.json`, `code.json`,
`code_mbpp.json`, `aime_greedy10.json` (10 greedy seeds), `aime_sampled.json` (n=16, temp 0.6).
