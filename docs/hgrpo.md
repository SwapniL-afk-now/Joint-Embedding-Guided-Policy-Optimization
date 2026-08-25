# H-GRPO: Hidden-State GRPO

GRPO's group-relative surrogate with the token policy replaced by a **group-normalised
distribution over hidden-state compatibility**. Runs as the auxiliary loss
`jepa.loss_type=h-grpo`; implementation in
[`core_algos.hgrpo_loss`](../verl/experimental/jepa_grpo/core_algos.py).

## Objective

$$
\mathcal{L} \;=\; \mathcal{L}_{\text{GRPO}} \;+\; \alpha\,\mathcal{L}_{\text{HGRPO}},
\qquad \alpha = 0.05
$$

One fused Adam step over the summed gradient (`jepa.fuse_optimizer_step=True`, enforced).

## Construction

For prompt $x_i$ with $G$ rollouts $y_{i1},\dots,y_{iG}$ and verifier labels
$c_{ij}\in\{0,1\}$:

**Frozen anchor** — the teacher's correct CoT, encoded by the live policy, stop-gradded:

$$
a_i \;=\; \operatorname{sg}\!\big[\operatorname{Normalize}\big(f_\theta(x_i,\,y_i^{T})\big)\big],
\qquad \nabla_\theta a_i = 0
$$

**Compatibility score** — last-content-token read ($k=0$, identity predictor):

$$
s_{ij} \;=\; a_i^\top z_{ij} \;=\; \cos(a_i,\,z_{ij})
$$

**Representation policy** — softmax over the group, augmented with a *virtual teacher
member* at $j=0$ with $s_{i0}=1$ (constant) and $c_{i0}=1$:

$$
q_\theta(j \mid x_i) \;=\; \frac{\exp(s_{ij})}{\sum_{k=0}^{G}\exp(s_{ik})}
$$

**Group-relative advantage** — from the verifier label, weighted by
$q^{\text{old}}=\operatorname{sg}[q_\theta]$:

$$
\bar c_i = \sum_j q^{\text{old}}_{ij} c_{ij},
\qquad
\sigma_i^2 = \sum_j q^{\text{old}}_{ij}\,(c_{ij}-\bar c_i)^2,
\qquad
A_{ij} = \frac{c_{ij}-\bar c_i}{\sigma_i+\epsilon}
$$

By construction $\sum_j q^{\text{old}}_{ij} A_{ij} = 0$.

**Loss** — the clipped ratio surrogate
$-\frac{1}{B}\sum_i\sum_j q^{\text{old}}_{ij}\min[r_{ij}A_{ij},\,
\operatorname{clip}(r_{ij},1\!\pm\!\varepsilon)A_{ij}]$ with
$r_{ij}=q_\theta/q^{\text{old}}$. Because $q^{\text{old}}$ is the detached value of the
**same** forward, $r\equiv 1$ identically, the clip is inactive by construction, and the
surrogate collapses algebraically to:

$$
\boxed{\;\mathcal{L}_{\text{HGRPO}}
\;=\; -\frac{1}{B}\sum_{i=1}^{B}\sum_{j=0}^{G} A_{ij}\; q_\theta(j\mid x_i)\;}
$$

**Score-level gradient:**

$$
\frac{\partial \mathcal{L}_{\text{HGRPO}}}{\partial s_{ij}}
\;=\; -\,q^{\text{old}}_{ij}\,A_{ij}
$$

Correct rollouts ($A_{ij}>0$) move toward the anchor; wrong rollouts ($A_{ij}<0$) move
away. For a mixed group the gap $\frac{d}{dt}(s^+ - s^-) > 0$ at the representation level.

## Why this form

| property | consequence |
| --- | --- |
| $q(s+\gamma) = q(s)$ exactly | The common-mode escape — raising every cosine together — is **not a direction in the objective**. This is precisely how `llm-jepa-triplet` failed (run `0ntqn3zc`: cos⁺ 0.8525→0.9393 and cos⁻ 0.8550→0.9445 in lockstep, margin ≈ 0, violation pinned at 1.00 for 170 steps). |
| readout direction **is** $a_i$ | No trainable readout $w$ — the BCE anticipation arm failed because its $w$ sat on the actor optimizer at lr 1e-6 and rotated 0.64° in 283 steps. |
| virtual teacher member | All-wrong groups ($c\equiv 0$) stay trainable: the teacher takes $A>0$ and every wrong rollout takes $A<0$, pushing their scores down. All-correct groups have $\sigma=0 \Rightarrow A\equiv 0$ and are dropped by the batch builder. |
| $A$ from the verifier label | Theta-independent — the PPO ratio corrects stale policy *estimates*; there is no estimate here to correct. |

## Invariants enforced by config validation

- `predictor_k = 0` — appending `[PRED]`/`[BAD_PRED]` by label (as the infoNCE/triplet
  path does) would be label leakage under a group softmax.
- `n_code = 0`, `teacher_cache_path` set — CoT students, CoT anchors.
- `fuse_optimizer_step = True` — two sequential Adam steps are not equivalent to one:
  Adam rescales each gradient by its own second moment, so $\alpha$ would stop
  controlling the relative contribution.

## Run

[`examples/jepa_grpo_trainer/long-context/run_hgrpo_math15b.sh`](../examples/jepa_grpo_trainer/long-context/run_hgrpo_math15b.sh)
— Qwen2.5-Math-1.5B-Instruct, DeepScaleR, 2 GPUs. Gates: grad-ratio ≈ 0.2 (step 25);
`jepa/margin > 0`, `jepa/violation < 1` (step 50, the falsifier); `rep/fixed_auroc_cw`
above the plain-GRPO 0.7514 (step 100); val pass@1 slope vs baseline (≥ 200 steps).

Tests: [`tests/experimental/jepa_grpo/test_hgrpo_on_cpu.py`](../tests/experimental/jepa_grpo/test_hgrpo_on_cpu.py)
— common-mode invariance (with the triplet loss as the failing contrast), monotone
gap growth, all-wrong-group signal via the teacher member, anchor stop-grad, ragged
groups, builder→loss contract.
