# Advantage-Aligned JEPA: Fixing the Gradient Orthogonality Problem

**Date:** 2026-08-02
**Runs audited:** `grpo-qwen-3b/atdmmu78` (JEPA, jepa-8cot-seed31415-666) vs `grpo-qwen-3b/fw7zf51z` (GRPO baseline)
**Scope:** `verl/experimental/jepa_grpo/` — final solution proposal

---

## 1. Problem statement

In the JEPA-GRPO run (`atdmmu78`) the auxiliary JEPA gradient is **near-orthogonal** to the
policy gradient:

- `jepa/grad_cos_with_policy ≈ 0` the whole run (mean +0.005, late phase only +0.03),
- `jepa/grad_ratio` (JEPA grad norm / policy grad norm) sits at **1.0–1.9×** after warmup
  (up to 11× during warmup),
- consequence: the aux term acts as a large orthogonal gradient that dilutes the effective
  policy update under Adam (JEPA's `actor/pg_loss` is 2–3× lower than baseline at matched
  steps), and the JEPA objective itself plateaued (`llm_jepa_loss` stuck at ~0.17,
  `cos_cot` stuck at 0.90 since step ~150).

The run tracks the baseline but trails by ~0.3–0.5 pp on validation at matched steps instead
of boosting it.

### Root cause (code-level)

**Run provenance (verified):** the JEPA arm was launched via
`grad_direction_probe/scripts_sc666/sc_jepa_4cot4code_400.sh` (tmux session `jepa`, GPU 0,
ray head `127.0.0.1:37379`) → `examples/jepa_grpo_trainer/qwen2.5-math-1.5b/run_drgrpo_jepa.sh`
with `JEPA_LOSS_TYPE=llm-jepa`, `N_COT=8`, `N_CODE=0`, `jepa.self_code_targets=True`,
`jepa.teacher_cache_path=""`, `jepa.code_teacher_cache_path=""`, `ALPHA=0.1`,
`jepa.paper_sigreg_lambda=0.1`, `jepa.fuse_optimizer_step=True`, full FT (`lora_rank=0`).

**There is NO teacher response in this run.** The teacher caches are empty
(`teacher_cache_path=""`; the script only checks them for loss types ≠ `llm-jepa`), and
`self_code_targets=True` makes the target the anchor's **own** rollout:

- **Anchor** `i` = prompt-only row (correct rollouts only, `jepa_anchor_set=correct`,
  ≤ `max_anchors_per_prompt`=2 per prompt); the worker appends `k` [PRED] tokens after the
  last prompt token → `p_i = Pred(Enc(prompt_i))` (`ray_trainer.py::_build_jepa_batch_llm_jepa`,
  `worker.py` llm-jepa path).
- **Target** = the anchor's own response text (response-only, assistant-prefixed), encoded
  and detached → `z_i = sg(Enc(response_i))`, gathered via `tgt_cot_idx`.
- **Loss** = `core_algos.py::llm_jepa_paper_loss`:
  `L = mean(1 − cos(p_i, z_i)) + λ·SIGReg(p)`, **flat mean over correct anchors** — no
  per-sample weighting, no wrong-rollout contrast, no group/advantage information anywhere.

Meanwhile the policy gradient (`trainer.py::_attach_dr_grpo_advantages`) weights the same
backbone by **group-relative advantage** `Â_i = R_i − mean(R_group)`.

| | Policy gradient | JEPA gradient (current) |
|---|---|---|
| Sample weighting | continuous advantage `Â_i` | binary "correct only" + flat mean |
| Target direction | make high-Â rollouts more likely | self-predict: prompt read → own response embedding |
| Dominant samples | high-Â rollouts | correct rollouts, uniformly (≤2/prompt) |
| Contrast | implicit via Â (up/down) | none — no negative/repel arm |

Two gradients over the same backbone with **disjoint support** (uniform-correct vs
advantage-weighted) and **no coupling between the pull strength and reward** → cos ≈ 0, and
the aux term can neither help nor be ignored. **The fix must change the loss, not the
gradients.**

---

## 2. Literature review

Our loss is a **self-predictive representation objective** (`p = Pred(Enc(prompt+[PRED]))`,
`z = sg(Enc(own response))`, cosine pull) combined with a sparse reward-driven policy
gradient. Five literature threads bear directly on making that combination boost rather
than dampen the policy:

### 2.1 Self-predictive (JEPA-family) auxiliary losses in RL — the design playbook

- **SPR — Self-Predictive Representations** (Schwarzer et al., ICLR 2021, arXiv:2007.05929):
  the canonical *self-predictive auxiliary loss in RL*: predict future latents with a
  **cosine loss**, **EMA target encoder**, no negatives, combined as
  `L_total = L_RL + λ·L_SPR`. The λ-weighting is set so the aux gradient is comparable to
  the RL gradient; this is exactly our architecture (predictor + detached target), and the
  paper's ablations show both EMA targets and augmentation matter for sample efficiency.
- **DeepMDP** (Gelada et al., ICML 2019): pure latent next-state prediction **collapses**
  without auxiliary reconstruction — the warning behind our plateau
  (`llm_jepa_loss` stuck at ~0.17, `cos_cot` stuck at 0.90: the predictor is
  approximately converged against a self-generated target, so the term provides almost no
  learning signal while still emitting a ~1.5–2× orthogonal gradient).
- **TD-JEPA** (arXiv:2510.00739, 2025): JEPA for zero-shot RL; finds that an
  **orthonormality/isometry regularizer on the representation is crucial to avoid
  collapse** — the same family as the run's SIGReg (`paper_sigreg_lambda=0.1`), confirming
  collapse-prevention is necessary but *not sufficient* for helping the policy.
- **LLM-JEPA** (Huang, LeCun, Balestriero, 2025, arXiv:2509.14252): in SFT the JEPA term
  "does not hinder" NTP — but the main task is dense next-token loss over the *same data*,
  so sample support trivially overlaps. In RL the main gradient is sparse and
  reward-weighted, so a uniform pull (correct-only anchors, flat mean) lives on a
  **different support** — the run's orthogonality is the empirical signature of that
  mismatch, not of JEPA per se.
- **SGI** (Schwarzer et al., NeurIPS 2021) and **CURL** (Srinivas et al., ICML 2020):
  representation aux losses that *help* RL when paired with the RL loss on the same data
  stream; **Contrastive RL** (Eysenbach et al., NeurIPS 2022) goes further and shows the
  representation pull and the value objective are the *same* learning signal when the
  positive/negative structure is reward-consistent.

Lesson for us: every successful self-predictive-in-RL recipe either (a) shares the RL
gradient's sample support, (b) couples the pull to reward, or (c) both. Ours does none.

### 2.2 Advantage weighting — the mechanism that couples an update to reward

- **AWR — Advantage-Weighted Regression** (Peng et al., NeurIPS 2019, arXiv:1910.00177):
  weight each sample by `exp(Â/β)`, derived as the solution of a **constrained policy-search
  problem**: the exponentiated advantage is the correct per-sample importance for turning a
  fitted/regressed update into an RL update. We are regressing representations, not actions,
  but the weighting identity is identical.
- **CRR — Critic Regularized Regression** (Wang et al., NeurIPS 2020): `min(1, exp(Â/β))`
  caps the weight — a stability guard relevant when groups are small (ours: G=8 rollouts).
- **SIL — Self-Imitation Learning** (Oh et al., ICML 2018): `max(0, R − V)` — *positive
  advantage only*; exactly the "push toward good samples, ignore the rest" structure we
  want for the pull arm.
- **SAW — State Advantage Weighting** (Lyu et al., 2022, arXiv:2210.04251): advantage-
  weights the update of a **prediction model** — the closest published analogue to
  advantage-weighting a JEPA-style predictive loss.
- **AWP lineage / RWR** (Peters & Schaal; Neumann & Peters): the same exponential-advantage
  family, all derived from the same constrained-MLE argument.

Lesson: weighting a supervised/predictive objective by advantage is a *derived* RL update,
not an ad-hoc trick — it makes the aux gradient follow the same reward landscape as the
policy gradient (shared support, monotone-correlated weights), which is precisely the
alignment condition our audit found missing.

### 2.3 Negative (repel) arm — hard negatives and contrastive structure

- **Contrastive Learning with Hard Negative Samples** (Robinson et al., ICLR 2021,
  arXiv:2010.04592): negatives that are *hard to distinguish from the anchor* carry most of
  the learning signal; hardness is controllable. For us the natural hard negatives are the
  **wrong/low-advantage rollouts of the same prompt** — maximally confusable with the good
  ones (same problem, similar text) yet with opposite reward.
- **InfoNCE/NCE practice** (Oord et al.; Wu et al.): pulling away from negatives is what
  prevents the representation from collapsing into a single point — the run's SIGReg
  only prevents *isotropy* collapse, not the *semantic* triviality of the target (own
  response ≈ own response). A repel arm graded by advantage gives the pull an actual
  direction.
- **SPR/BYOL's no-negative design** (Grill et al., 2020): works *without* negatives only
  because the EMA target is a *different, slower view* of the same input; our target is the
  same encoder's own output with no asymmetry, so we need either an EMA target or explicit
  negatives — the repel arm is the cheaper fix.
- **Triplet/margin losses** (Schroff et al., FaceNet 2015): margin `m` between positive and
  negative similarities — the form of Change 2.

Lesson: the wrong-answer arm is not decoration — it is the standard contrastive mechanism
(and its absence is why the current loss plateaus into a trivial predictor).

### 2.4 Combining the aux loss with the RL loss (weights, not surgery)

- **ALI / AuxiLearn** (Navon et al., ICLR 2021, arXiv:2007.02693): learn the aux weight via
  implicit differentiation so the aux *improves the main objective*; its baseline "gradient
  cosine similarity" weighting (Du et al., 2018) is the family the codebase already tried
  (`JEPA_PCGRAD`, `JEPA_KEEP_ALIGNED`, `JEPA_LAYER_RATIO`, `JEPA_GRAD_RATIO` in
  `worker.py`) — reweighting/rejecting an orthogonal gradient instead of making it aligned.
  **Rejected** (consistent with the user's constraint: no PCGrad-style surgery).
- **GradNorm** (Chen et al., ICML 2018): balance aux/main by gradient *magnitude* —
  the principled fix for the measured `grad_ratio ≈ 1.5–2×` (stage-2 Change 3).
- **"Loss is its own Reward"** (Shelhamer et al., ICLR 2017): aux losses in RL only help
  when gradient magnitudes are calibrated across tasks (Krähenbühl et al. calibration) —
  same lesson as GradNorm.

Lesson: the *weight* (α) is a legitimate knob and has a learned-solution literature; the
*direction* of the aux gradient must be fixed in the loss, not patched at merge time.

### 2.5 Synthesis — why this combination aligns the gradients

The policy gradient is `Σ_i Â_i · ∇log π_i` (support: high-Â rollouts). Change 1 makes the
JEPA pull `Σ_i w(Â_i) · ∇(1−cos)` with `w` monotone in `Â` over the **same** rollouts —
the two gradients are dominated by the same samples with correlated weights (the AWR
identity). Change 2 adds the contrastive structure that gives the pull its direction and
prevents the trivial predictor (hard-negatives literature). Change 3 controls the magnitude
ratio (GradNorm/ALI). No gradient projection anywhere: alignment is a *property of the
loss*, which is the only thing the literature supports as a stable fix.

---

## 3. Final solution

Three changes, in increasing order of ambition. Changes 1–2 are the core fix; change 3 is
optional stage 2.

### Change 1 — Advantage-weighted aggregation of the self-predictive pull (core fix) [IMPLEMENTED]

Replace the **correct-only + flat mean** aggregation in `ray_trainer._build_jepa_batch_llm_jepa`
+ `worker.py`'s llm-jepa path with **AWR-style group-normalized advantage weighting** (still
no teacher response — target stays inside the model's own sampled responses):

```
w_i = softmax(Â_i / τ) over the prompt's group        # τ auto = group advantage std (floor 1e-3)
align = Σ_i w_i · (1 − ⟨pred_i, sg(z_i)⟩) / Σ_i w_i   # z_i = Enc(own response) by default
```

- **Anchors = all sampled rollouts of the prompt** (ignoring `jepa_anchor_set` and
  `max_anchors_per_prompt`); the advantage decides only *how hard* each anchor is pulled.
- **Target (default `adv_target=best`): every anchor in a prompt is pulled toward the
  embedding of the group's highest-advantage response** — a single high-quality internal
  target per prompt, with no teacher text. `adv_target=self` (anchor's own response) is
  available as a knob.
- **`adv_tau=0.0` auto**: per-group std of that group's advantages, floored at 1e-3;
  single-rollout groups get τ=1. Fixed `adv_tau>0` also supported.
- Config: `jepa.adv_weighting=True jepa.adv_target=best jepa.adv_tau=0.0`.

**Why this aligns the gradients:** the policy gradient is dominated by high-Â rollouts; the
JEPA pull is now dominated by the *same* rollouts, with weights monotone in the *same* `Â`.
Shared support ⇒ the two gradients over the backbone are no longer dot-products of disjoint
random vectors. In groups where all rollouts are correct (all Â ≈ 0) the softmax is
near-uniform and the pull is weak — JEPA shuts off exactly where the policy gradient does.
Targeting the group-best response additionally removes the trivial self-collapse of the old
`sym(Enc)` read and points the predictor at the very response the policy is being pushed
toward.

**Implementation (done):**
- `ray_trainer.py::_build_jepa_batch_llm_jepa` — reads the per-sequence Dr.GRPO advantages
  written by `compute_advantage` (Step 2.7) into `batch.batch["advantages"]`; emits per-anchor
  `anchor_adv_weight` (softmax(Â/τ) per group), `anchor_neg_weight`, `neg_tgt_pos`; selects
  the group-best response row as the shared target (`tgt_cot_idx`). Requires
  `self_code_targets=True`.
- `worker.py` llm-jepa path — passes the shipped weights + neg target into the loss.
- `core_algos.py::llm_jepa_paper_loss` — accepts `weights`/`neg_weights`/`neg_target`/
  `neg_margin`; weighted pull `(ell·w).sum()/w.sum()`; falls back byte-for-byte to the flat
  mean when the args are absent.

### Change 2 — Advantage-conditioned push-away (the repel half; previously absent) [IMPLEMENTED]

The old loss has **no contrast at all** (correct anchors only, no negatives). Add a repel arm
over the same joint forward: anchors with **strictly negative** advantage (`Â < 0`) are pushed
**away from their own response embedding** by a triplet hinge, weighted by
`softmax(−Â_i/τ)` (zeroed on `Â ≥ 0` rows):

```
align_neg = Σ_i w_neg_i · max(0, m − ⟨pred_i, sg(z_i)⟩)    # margin m = jepa.neg_margin (0.2)
```

The bad anchors' own response rows are added to the (deduplicated) target block and gathered
via the new `neg_tgt_pos`; only rows with `w_neg > 0` contribute (and only when
`jepa.neg_pull=True`). This gives the loss explicit signed semantics — the model learns
**which hidden states to push toward (high advantage) and which to pull away from (low
advantage)** — graded by advantage instead of a binary correct/wrong filter.

### Change 3 — Learned aux weight (optional stage 2)

Replace the fixed `alpha=0.1` with a learned weight via **ALI-style implicit differentiation**
or **GradNorm** (Chen et al., ICML 2018). This fixes the `grad_ratio ≈ 1.5–2×` problem in a
principled way (no manual norm-matching), keeping the aux term at a bounded fraction of the
policy gradient.

### Explicitly excluded

- PCGrad / CAGrad / gradient gating / per-layer gating (`JEPA_PCGRAD`,
  `JEPA_KEEP_ALIGNED`, `JEPA_LAYER_RATIO`) — symptom patches already tried in this codebase.
- Global or per-layer norm-matching (`JEPA_GRAD_RATIO`) — treats magnitude, not direction.

---

## 4. Why this aligns the gradients

After the change:

```
g_jepa   = Σ_i w(Â_i) · ∇_θ ℓ_i          (w monotone increasing in Â)
g_policy = Σ_i Â_i · ∇_θ log π_i
```

1. **Shared support.** Both gradients are dominated by the *same* high-advantage samples of
   the *same* batch, with *correlated* weights (both monotone in Â). The cosine between them
   becomes positive by construction instead of being the dot product of disjoint random
   vectors.

2. **Per-sample directional agreement.** For a high-Â anchor, the JEPA pull makes the
   prompt's [PRED] read predict that rollout's own response representation; the policy
   gradient raises that same rollout's likelihood. Both reinforce the encoding of the same
   "good" response from the same prompt state, so ⟨∇ℓ_i, ∇log π_i⟩ > 0 for the dominant
   samples.

3. **Self-normalizing support.** In groups where everyone is correct (all Â ≈ 0) the softmax
   is near-uniform and the JEPA term shuts off — **exactly where the policy gradient shuts
   off**. In groups with partial success, both objectives focus on the same samples. The aux
   term stops being noise and becomes a booster.

4. **The target moves with the policy.** Because the target is the policy's own response
   embedding (stop-grad), the JEPA loss tracks the improving policy's representation space
   instead of stalling against a static reference — the pull keeps providing signal.

---

## 5. Expected effects

| Metric | Before | After (expected) |
|---|---|---|
| `jepa/grad_cos_with_policy` | ≈ 0 (mean +0.005) | clearly positive (> 0.1) and sustained |
| `jepa/llm_jepa_loss` | plateaued at ~0.17 since step 150 | keeps decreasing (target moves with policy) |
| `actor/pg_loss` vs baseline | 2–3× lower at matched steps | ≈ baseline or better |
| Val (5-seed mean, matched steps) | baseline −0.3…−0.5 pp | ≥ baseline |
| `grad_ratio` | 1.5–2× | bounded (stage 2: learned α) |

---

## 6. Verification plan

1. **Gradient alignment:** `jepa/grad_cos_with_policy` should turn clearly positive within
   the first ~50–100 steps and stay positive.
2. **Objective health:** `llm_jepa_loss` / `cos_cot` should not plateau at a static floor.
3. **Head-to-head:** same model/data/reward/eval-seeds as the audit pair, matched-step
   comparison on aime24/25/26 + amc23, 5-seed mean.
4. **Ablations (cheap):**
   - Change 1 only vs Change 1+2 (is the negative pull worth it?).
   - Fixed τ vs τ = group reward std.
   - All-anchor vs correct-only anchor sets (does advantage weighting make wrong anchors
     safe to include?).

---

## 7. Files touched (minimal)

| File | Change |
|---|---|
| `ray_trainer.py` | `_build_jepa_batch_llm_jepa`: read `batch.batch["advantages"]` (from Step 2.7 `compute_advantage`); all-rollout anchors; group-best response target row (`adv_target=best`) or own-response (`self`); per-group softmax weights `softmax(Â/τ)` (τ auto = group std, floor 1e-3) + `softmax(−Â/τ)` zeroed on `Â ≥ 0`; new batch keys `anchor_adv_weight`, `anchor_neg_weight`, `neg_tgt_pos` |
| `worker.py` | llm-jepa path: plumb the shipped weights + `neg_tgt_pos` (own-response repel target) into `llm_jepa_paper_loss` |
| `core_algos.py` | `llm_jepa_paper_loss`: optional `weights` (weighted pull `Σ w·ell / Σ w`), `neg_weights`/`neg_target`/`neg_margin` (triplet repel arm), new metrics `jepa/repel_loss`, `jepa/n_neg_anchors`, `jepa/adv_weight_mean`, `jepa/neg_weight_mean`; byte-identical flat-mean fallback |
| `config_ray.py` | `jepa.adv_weighting: bool=False`, `jepa.adv_target: 'best'\|'self'`, `jepa.adv_tau: 0.0` (auto), `jepa.neg_pull: bool=False`, `jepa.neg_margin: 0.2` + validation |
| `grad_direction_probe/scripts_sc666/sc_jepa_adv_666.sh` | new 666-step arm enabling `adv_weighting` + `neg_pull` (head-to-head vs `sc_jepa_4cot4code_400.sh`) |
