# sc_dec / sc_fus / sc_fixed run scripts (Aug 1, 2026)

All runs launch `run_drgrpo_jepa.sh` from the qwen2.5-math-1.5b training dir.

| Run | Wrapper | Extra args | Checkpoint dir | Wandb run | Log |
|---|---|---|---|---|---|
| sc_dec | `sc_dec.sh` | `jepa.self_code_targets=True jepa.paper_sigreg_lambda=0.1` | `selfcode_sigreg_decoupled_r01_666` | `run-20260801_015812-uhm15vt6` | `log_sc_dec666.log` |
| sc_fus | `sc_fus.sh` | same + `jepa.fuse_optimizer_step=True` | `selfcode_sigreg_fused_r01_666` | `run-20260801_015816-b1w9lonx` | `log_sc_fus666.log` |
| sc_fixed | `sc_fixed.sh` | same as sc_fus; env `N_COT=4 N_CODE=4 JEPA_LOSS_TYPE=llm-jepa ALPHA=0.1 AUTO_OFF_MIN_COS=0.5` | `selfcode_sigreg_fixed_r01_666` | — | `log_sc_fixed666.log` |

## sc_fixed — fix pass 1 (F1+F2+F3A+F5), launched ~13:00 Aug 1

Replaces the latched sc_dec/sc_fus pair. Code changes (all in
`verl/experimental/jepa_grpo/`):
- **F1** (`ray_trainer.py` `_build_jepa_batch_llm_jepa`): anchor rows are now
  `[prompt; response]` (paper view pairing, `_anchor_ids`), targets stay
  response-only. Previously anchors were response-only + constant assistant
  header, which removed all per-problem signal.
- **F2** (`worker.py` llm-jepa `_loss_fn`): `tgt = joint_emb[_N:].detach()` —
  zero gradient through the target branch.
- **F3A** (`core_algos.py` `llm_jepa_paper_loss`, `center=True` default): cosine
  computed on per-side batch-mean-centered, renormalized embeddings. Raw cosines
  still logged as `jepa/cos_pred_target_raw` / `jepa/cos_code_raw`.
- **F5** (env): `ALPHA=0.1` (was 0.03), `AUTO_OFF_MIN_COS=0.5` (was 0.0).
- Auto-off cot signal repointed to `jepa/cos_cot_centered` (ray_trainer.py
  `_tracked_signals`); `jepa/n_unique_targets` now emitted from meta_info.

### Gate metrics (from plan)
| Metric | latched baseline | sc_fixed target |
|---|---|---|
| `retrieval_top1` (chance 0.029, N~35) | 0.025-0.05 | ≥0.1 by step 100, ≥0.3 by 250 |
| `cos_pred_target_centered` | ~0.02 flat | ≥0.1 by step 250 |
| `cos_all_pairs` | 0.83-0.91 | ≤0.6 |
| `effective_rank_norm` | 0.042 | no collapse (ranks ~10-14/40) |
| `off_cot` | latched step 42 | must NOT latch |
| `code/pass_at_1` | 0.562 @250 | no regression |

Smoke test (2 runs, `smoke_fixed_test`, 2 steps) validated the code path before
launch; smoke checkpoints deleted. Baselines for A/B: metrics jsonl + wandb runs
uhm15vt6 / b1w9lonx (+ checkpoints at global_step_250).
