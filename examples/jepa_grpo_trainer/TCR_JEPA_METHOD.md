# Teacher-Correct Representation Alignment as a JEPA-Style Auxiliary Objective for RL of Reasoning LLMs

## Abstract

We augment reinforcement learning (RL) of a reasoning policy with an auxiliary
*representation-alignment* objective. Alongside the standard reward-driven policy
gradient, the policy is trained so that the internal representation it forms at the point
of prediction is drawn toward the representation of a *verified-correct* solution to the
same problem. Correct solutions are produced once, offline, by a teacher and verified by
the same correctness checker that defines the RL reward; **only their text is stored.** At
training time no teacher is present: the alignment target is recomputed *online* by the
current policy itself, in a stop-gradient target branch, following the LLM-JEPA
paired-view design. We refer to this objective as **Teacher-Correct Representation
alignment in a Joint-Embedding Predictive Architecture (TCR-JEPA)**. Crucially, the
target is expressed in the policy's *own*, *current* representation space, so the
alignment is a within-space attraction that requires neither a learned projection nor a
frozen reference encoder, and it never goes stale.

---

## 1. Motivation

A policy trained purely from a scalar reward receives no information about *what a correct
internal state looks like* — only about whether its emitted text happened to score. We
hypothesize that supervising the policy's latent representation, in addition to its
behavior, accelerates the acquisition of correct reasoning. The supervision signal is the
latent of a known-correct solution, expressed in the policy's own representation space.

The method imposes two simultaneous objectives on the policy:

* a **behavioral** objective — the RL policy gradient (Dr.GRPO), which rewards text that
  solves the task; and
* a **representational** objective — TCR-JEPA alignment, which shapes the latent formed at
  the point of prediction to resemble that of a verified-correct solution.

The central design decision separating this work from our earlier fixed-latent variant
(Section 6.3) and from teacher-hidden-state distillation (Section 6.2) is that the target
representation is **recomputed online by the current policy**, not cached. The teacher is
a *text oracle used once offline*; it is never loaded during RL and never supplies hidden
states.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $x$ | a problem (prompt) |
| $\mathcal{V}(\cdot)$ | correctness verifier; also defines the RL reward |
| $\tau$ | strong teacher model (offline only) |
| $y^+$ | a verified-correct solution *text* for $x$ |
| $\pi_\theta$ | current policy (the model being trained) |
| $f_\theta(\cdot)$ | the policy's encoder; returns final-layer hidden states |
| $\langle\mathrm{predict}\rangle$ | dedicated prediction token appended to the rollout view |
| $\operatorname{sg}[\cdot]$ | stop-gradient |
| $\operatorname{normalize}(\cdot)$ | $\ell_2$ normalization to the unit sphere |
| $G$ | rollout group size per problem |
| $R_i$ | scalar reward of the $i$-th rollout |

---

## 3. Method Overview

The method has two stages. Stage I is performed once, offline; Stage II is the RL loop.

* **Stage I — Offline verified-correct solution collection.** A teacher samples candidate
  solutions; the verifier filters them; only correct *texts* $(x, y^+)$ are retained. No
  hidden states or latent vectors are cached.
* **Stage II — Online RL with JEPA-style correct-solution representation alignment.** The
  current policy generates rollouts under Dr.GRPO. For each rollout we build two views — a
  **prediction view** of the policy's own rollout and a **target view** of a stored
  verified-correct solution — *both encoded by the current policy*, and pull the
  prediction representation toward the (stop-gradient) target representation.

---

## 4. Stage I — Offline Verified-Correct Solution Collection

For each problem $x$ in the training set, the teacher $\tau$ samples $M$ candidate
solutions $\{\tilde y_1,\dots,\tilde y_M\}\sim\tau(\cdot\mid x)$. Each candidate is scored
by the *same* correctness checker used for RL rewards:

$$
\mathcal{C}(x) = \{\, \tilde y_m : \mathcal{V}(x,\tilde y_m)=1 \,\}.
$$

We retain the verified-correct set as **text** and store the pairs $(x, y^+)$ for
$y^+\in\mathcal{C}(x)$. Problems with $\mathcal{C}(x)=\varnothing$ are either dropped from
the auxiliary objective or back-filled by additional teacher sampling. **No hidden states,
no latent vectors, and no teacher activations are cached** — only solution strings. This
is the single offline artifact consumed by Stage II.

When a problem has multiple stored $y^+$, a deterministic *cycle* match (or a random
match) assigns one target per anchor at each step.

---

## 5. Stage II — Online RL with JEPA-Style Alignment

### 5.1 Rollout

For problem $x$, the current policy samples a rollout

$$
y \sim \pi_\theta(\cdot \mid x).
$$

### 5.2 Prediction branch

The prediction branch receives the policy's *own* rollout, terminated by the dedicated
prediction token, and reads the final-layer hidden state **at the $\langle\mathrm{predict}\rangle$ token**:

$$
p_\theta = \operatorname{normalize}\!\big(f_\theta(\,[\,x,\,y,\,\langle\mathrm{predict}\rangle\,]\,)\big).
$$

This branch carries gradient: it is the representation we shape.

### 5.3 Correct-solution target branch

The target branch receives a stored verified-correct solution and reads the final-layer
hidden state **at the last real token of $y^+$**:

$$
z^+_\theta = \operatorname{sg}\!\Big[\operatorname{normalize}\!\big(f_\theta(\,[\,x,\,y^+\,]\,)\big)\Big].
$$

Two properties are essential:

1. **Same encoder, current weights.** $z^+_\theta$ is produced by $f_\theta$ — the
   *current* policy — not by a frozen reference or a cache. The target therefore lives in
   the policy's present representation space and tracks it through training.
2. **Stop-gradient.** $\operatorname{sg}[\cdot]$ blocks gradient flow through the target
   branch, so the target acts as a (locally) fixed anchor within a step while remaining
   globally non-stale across steps. This is the LLM-JEPA paired-view pattern. **No EMA
   target encoder is used**; an EMA encoder is an optional variant (Section 7), not part
   of the base method.

### 5.4 Alignment loss

Alignment is the negative cosine similarity between the two unit-norm views:

$$
L_{\text{align}} = 1 - \langle p_\theta,\, z^+_\theta \rangle.
$$

Because both views are produced by the same current encoder, this is a *within-space*
attraction; no cross-encoder projection or dimensionality bridge is required.

### 5.5 Anti-collapse regularization

A pure attraction toward stop-gradient targets admits a trivial constant-embedding
solution. We add an anti-collapse regularizer over the prediction embeddings — SIGReg
(LeJEPA's sketched isotropic-Gaussian regularizer via the Epps–Pulley statistic over
random projection directions), or an equivalent variance/covariance term:

$$
L_{\text{TCR-JEPA}} = (1-\lambda)\,L_{\text{align}} + \lambda\,L_{\text{SIGReg}}.
$$

$L_{\text{SIGReg}}$ pushes the batch of $p_\theta$ toward an isotropic distribution on the
sphere, preventing representational collapse that $L_{\text{align}}$ alone would permit.

---

## 6. Behavioral RL Objective (Dr.GRPO)

The behavioral objective is unchanged from Dr.GRPO. For each problem the policy samples a
group of $G$ rollouts with rewards $\{R_i\}$. We use **group-relative advantages without
standard-deviation normalization**:

$$
\hat{A}_i = R_i - \frac{1}{G}\sum_{j=1}^{G} R_j.
$$

With token-level importance ratio
$r_{i,t} = \pi_\theta(y_{i,t}\mid x, y_{i,<t}) / \pi_{\theta_{\text{old}}}(y_{i,t}\mid x, y_{i,<t})$,
the clipped policy-gradient loss over the set $\mathcal{T}$ of (rollout, token) pairs is

$$
L_{\text{PG}}(\theta) = -\frac{1}{|\mathcal{T}|}\sum_{(i,t)\in\mathcal{T}}
\min\!\Big(r_{i,t}\hat{A}_i,\; \operatorname{clip}(r_{i,t},\,1-\epsilon,\,1+\epsilon)\,\hat{A}_i\Big).
$$

An optional KL penalty anchors the policy to a frozen reference $\pi_{\text{ref}}$:

$$
L_{\text{actor}} = L_{\text{PG}} + \beta\,\mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big).
$$

The conceptual full objective combines behavior and representation:

$$
\boxed{\,L_{\text{total}} = L_{\text{actor}} + \alpha\,L_{\text{TCR-JEPA}}\,}.
$$

$\alpha$ trades off the auxiliary alignment against the RL signal. In practice the two
terms are optimized as coordinated backward passes sharing the policy parameters; because
$L_{\text{align}}$ is averaged over a comparatively small pool of valid anchor–target
pairs while $L_{\text{PG}}$ is token-mean over the full batch, $\alpha$ must be set to
restore gradient-norm parity between the two.

---

## 7. Anchor Modes

The prediction view requires a rollout $y$ to serve as the *context* whose
$\langle\mathrm{predict}\rangle$ representation is aligned. Which rollouts qualify defines
three modes:

1. **Correct anchors (default).** Only *successful* policy rollouts ($R_i=1$) are used as
   prediction contexts. The model is asked to make the latent of a self-generated correct
   trajectory resemble that of a verified-correct solution — a consistency pull among
   correct states.

2. **Wrong anchors (post-trajectory latent correction).** *Failed* rollouts are used as
   contexts, but only the final $\langle\mathrm{predict}\rangle$ representation is aligned
   to the verified-correct target. This is explicitly **not** making wrong reasoning look
   correct token-by-token; the wrong trajectory's tokens are untouched. It is a
   *post-trajectory* correction: after having produced a flawed attempt, the model is
   trained so that the representation it forms at the prediction point still points toward
   a correct solution's latent — a "recover the right conclusion despite the flawed path"
   signal.

3. **Both anchors.** Correct and wrong anchors are both used, preferably with
   **problem-level reward-stratified aggregation** so that easy problems (mostly correct
   rollouts) and hard problems (mostly wrong rollouts) contribute comparably rather than
   being dominated by whichever stratum is more populous.

---

## 8. Relationship to Prior Approaches

The defining choices of TCR-JEPA — (i) the target is a *verified-correct solution*, (ii)
encoded *online by the current policy*, (iii) under *stop-gradient*, (iv) as a
*representation-level* target rather than a token target — distinguish it cleanly from
nearby methods.

### 8.1 vs. SFT (supervised fine-tuning on correct solutions)
SFT trains directly on the *tokens* of fixed correct outputs, i.e. a maximum-likelihood
imitation target. TCR-JEPA never imitates $y^+$ at the token level. The policy is still
trained on its *own* RL-generated rollouts under Dr.GRPO; $y^+$ enters only as a
*representation-level target view*. The behavioral update and the on-policy distribution
are preserved.

### 8.2 vs. OPRD (online policy representation distillation)
OPRD aligns *student* hidden states to *teacher* hidden states **on the same
student-generated rollout** — the teacher must be loaded online and supply activations.
TCR-JEPA uses **no teacher hidden states online**. It aligns a *policy-generated rollout
view* to a *verified-correct-solution view*, both encoded by the **current policy** with
stop-gradient. The teacher contributes text offline and is absent at training time.

### 8.3 vs. offline representation distillation / fixed-latent TCR
Offline representation distillation (and our original fixed-latent TCR) caches $z^+$ once,
using a *frozen reference encoder*, and aligns to that frozen vector throughout training.
As the policy's representation space drifts, those cached coordinates become **stale** and
the alignment chases a target that no longer lives in the policy's current geometry. The
corrected TCR-JEPA caches only the *text* $y^+$ and recomputes $z^+_\theta$ online with the
current policy under stop-gradient, eliminating the stale latent-coordinate mismatch.

### 8.4 vs. Dr.GRPO
Dr.GRPO is the *behavioral backbone* of TCR-JEPA and is left intact (Section 6). TCR-JEPA
adds the auxiliary $\alpha L_{\text{TCR-JEPA}}$ term; with $\alpha=0$ it reduces exactly to
Dr.GRPO.

### 8.5 Comparison table

| Method | Target signal | Target encoder | Online teacher? | Target freshness | Updates policy via |
|---|---|---|---|---|---|
| **SFT** | correct output **tokens** | — (token CE) | no | n/a | MLE on fixed tokens |
| **Dr.GRPO** | scalar reward only | — | no | n/a | on-policy PG (no std-norm) |
| **OPRD** | teacher **hidden states** on student rollout | teacher (online) | **yes** | fresh (teacher) | PG + hidden-state distill |
| **Offline rep. distill / fixed-latent TCR** | cached latent $z^+$ | frozen reference (offline) | no | **stale** (cached) | PG + align to fixed vector |
| **TCR-JEPA (ours)** | verified-correct **solution view** | **current policy** (online, sg) | no | **fresh** (recomputed) | Dr.GRPO PG + JEPA align |

---

## 9. Summary

TCR-JEPA keeps Dr.GRPO as the behavioral objective and adds a JEPA-style representational
objective in which the policy's prediction-point latent is pulled toward the latent of a
verified-correct solution. The teacher is a purely offline *text* oracle; the alignment
target is recomputed online by the current policy under stop-gradient, with an
anti-collapse regularizer, so the supervision is always expressed in the policy's own,
current representation space. This removes the stale fixed-latent failure mode of the
original TCR while retaining its core hypothesis: that supervising *what a correct internal
state looks like*, not just whether the output scored, accelerates the acquisition of
correct reasoning.
