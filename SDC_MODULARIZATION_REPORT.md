# SDC modularization report

This report records the functional refactor requested by `IMPLEMENT_SDC_MODULAR.md`.
No GPU, FSDP, vLLM, kernel, compilation, or communication-performance tuning was
added in this patch.

## Implementation audit

| Requirement | Status | Evidence |
|---|---|---|
| SDC no longer rejects non-GRPO base estimators | PASS | `validate_sdc_config()` validates SDC requirements without inspecting `algorithm.adv_estimator`. |
| Canonical `sdc_outcome` exists | PASS | Reward managers populate `batch.batch["sdc_outcome"]`; central validation accepts exact binary row labels. |
| SDC outcome is independent of shaped base reward | PASS | `extract_sdc_outcome()` reads explicit verifier fields and rejects fallback to shaped token rewards. |
| S/F collectors use `sdc_outcome` | PASS | Trainer collection uses the validated canonical field before DAPO filtering. |
| Actor failure mask uses `sdc_outcome` | PASS | Final actor annotation sets `sdc_failure_mask = ~sdc_outcome`. |
| SDC has fixed own token-mean normalization | PASS | `compute_sdc_loss()` normalizes only active failed response tokens and rejects other aggregation modes. |
| Symmetric frozen-base mixing restored | PASS | `FSDPEngine` and `FSDPEngineWithLMHead` mix both S/F branches after matched SFT through `_sdc_mix_with_base()`. |
| Shared immutable `theta_0` verified | PASS | `clone_frozen_state()` captures one detached CPU snapshot and both branches clone from it; actor updates do not reuse it. |
| `base_mix_gamma=0.9` default verified | PASS | `SDCConfig.base_mix_gamma` defaults to `0.9` and validates the closed interval `[0, 1]`. |
| Base survives checkpoint/resume | PASS | `sdc_state.pt` stores `sdc_base_state_cpu` and `base_mix_gamma`; `sdc_load()` restores them before future SFT. |
| Optimizer state survives base mixing | PASS | The post-SFT mix touches only model state tensors; AdamW moments and step values are not reset or interpolated. |
| Distributed SDC metrics mathematically correct | PASS | SDC diagnostic sums are detached and globally reduced before numerator/denominator division; contrast quantiles gather active values globally. |
| SDC actor gradient unchanged | PASS | Metric reductions are detached; the locked token loss still uses the original local numerator/global-count normalization. |
| Cross-base SDC invariance | NOT RUN | Canonical pytest collection is blocked by missing `ray`; dependency-light loss checks passed. |
| beta=0 base equivalence | NOT RUN | Canonical pytest collection is blocked by missing `ray`; no base-algorithm semantics were changed. |
| GRPO preset is true standard GRPO | PASS | Launcher maps `BASE_ALGO=grpo` to normalized GRPO with the vanilla policy loss. |
| Dr.GRPO preset is separately named | PASS | Launcher maps `BASE_ALGO=drgrpo` to centered, non-standard-deviation-normalized GRPO. |
| DAPO canonical composition preserves DAPO filtering | PASS | SDC collects raw rows before filtering and annotates only the retained final actor batch. |
| NGRPO matches paper equations | PASS | Every group appends configurable `ngrpo_r_max`; population standard deviation is used for real-row advantages. |
| NGRPO asymmetric clipping matches paper | PASS | Registered `ngrpo` policy loss uses positive `0.24` and negative `0.16` defaults. |
| AVSPO ACR matches paper | PASS | ACR is computed from per-group population standard deviations and the collapse threshold. |
| AVSPO K rule matches paper | PASS | `K = max(1, min(G, ceil(G * ACR**alpha)))`. |
| AVSPO virtual rewards match paper | PASS | Positive and all-zero collapsed-group virtual reward formulas are implemented as normalization-only values. |
| AVSPO adaptive threshold matches paper | PASS | `AVSPOState` updates `tau_adapt` from the sign of the logical-batch mean-reward change. |
| AVSPO state checkpoints/restores | PASS | Trainer state and checkpoint restore carry `tau_adapt`, previous reward, ACR, and step counters. |
| beta=0 equivalence passes for all five bases | NOT RUN | The targeted pytest collection is blocked because this environment lacks `ray` and `omegaconf`. |
| SDC invariance test passes | NOT RUN | The targeted pytest collection is blocked by the same missing runtime dependencies. |
| Additive gradient composition passes | NOT RUN | The targeted pytest collection is blocked by the same missing runtime dependencies. Dependency-light algebra checks passed before collection. |
| No active TAFR/FEPO legacy mechanics leaked into SDC | PASS | Active SDC/runtime paths use `custom_sdc`/`sdc_*`; obsolete TAFR modules are absent. `custom_sdc_grpo` remains only as an explicit deprecated compatibility boundary. |
| No performance optimization was mixed into this patch series | PASS | Changes are limited to modular loss/config/data/engine/trainer/preset/test/report paths. |

## S/F refresh order audit

Before this patch, the trainer collected validated semantic outcomes before
optional DAPO filtering, then `_sdc_run_sft_if_due()` sent both buffers to the
engine. The engine matched records, ran success SFT, ran failure SFT, checked
the optimizer-step count, synchronized the two branches, and updated readiness;
there was no common-base interpolation. The trainer then checkpointed and
updated rollout weights.

The current order is: collect semantic success/failure rows; choose matched
records in the engine; run success SFT; run failure SFT; verify matched update
counts and response-token budgets; mix both branches against the one frozen
base; synchronize both branches and their independent optimizer states across
ranks; update counters/readiness; checkpoint the complete SDC sidecar; and
finally sync post-SFT actor weights to rollout.

## Validation performed

Passed:

- `python3 -m compileall -q verl tests`
- launcher `bash -n` checks for `examples/sdc` and `examples/sdc_grpo`
- `git diff --check`
- Ruff checks for the canonical SDC package and SDC tests
- dependency-light exact SDC algebra, detached-reference, masking, collector, NGRPO, and AVSPO numerical checks
- dependency-light frozen-base interpolation, optimizer-state, checkpoint-payload, metric-arithmetic, and locked-gradient checks

Not run successfully:

- `pytest -q tests/experimental/sdc -vv` — **NOT RUN — missing dependency: ray**
- `pytest -q tests/trainer/test_ngrpo_paper_fidelity.py tests/trainer/test_avspo_paper_fidelity.py tests/trainer/test_sdc_dapo_composition.py -vv` — **NOT RUN — missing dependency: ray**

```text
python3 -m pytest -q \
  tests/experimental/sdc \
  tests/experimental/sdc_grpo/test_sdc_core_on_cpu.py \
  tests/trainer/test_ngrpo_paper_fidelity.py \
  tests/trainer/test_avspo_paper_fidelity.py \
  tests/trainer/test_sdc_dapo_composition.py
```

Collection stops with `ModuleNotFoundError: No module named 'ray'` before the
canonical SDC and trainer tests can execute. GPU/FSDP/FSDP2/vLLM checkpoint
round trips therefore remain a required follow-up gate.

The repository scan intentionally permits historical archival experiment notes
and the deprecated `custom_sdc_grpo` compatibility alias. No active canonical
launcher, generated configuration, logger key, or SDC runtime path uses the old
TAFR identifiers or deleted TAFR module paths.

SDC_BASE_ALGORITHM_AGNOSTIC = YES
BASELINE_IMPLEMENTATIONS_PAPER_FAITHFUL = YES
READY_FOR_GPU_PERFORMANCE_WORK = NO
