# How SDC-TR Learns from Successes and Failures

This guide explains the training loop in plain language. It focuses on the
current `sdc_tr` setup, including the zero-advantage repair used by the
`drgrpo_a0.5_dg0.5` run.

## The short version

The policy (the model being trained) generates several answers for each math
problem. A verifier labels each answer as **success** or **failure**.

From those answers, SDC maintains two sidecar teacher models:

- a **success teacher**, trained only on correct answers;
- a **failure teacher**, trained only on incorrect answers.

The teachers are not two models that generate new answers. They are two copies
of the policy used to ask a comparison question:

> For this token, does a success-trained model or a failure-trained model assign
> higher probability?

The actor then uses that comparison together with PPO/GRPO:

- increase probability for useful behavior;
- decrease probability for failure-like behavior;
- keep every update inside PPO's trust-region safeguards.

## 1. The three kinds of model involved

It is useful to separate these roles:

| Model | Role |
|---|---|
| **Actor / policy** `pi_theta` | Generates answers and receives gradient updates from RL. |
| **Old policy** `pi_old` | Snapshot used by PPO to measure how far the actor moved. |
| **Success/failure teachers** `pi_S`, `pi_F` | Sidecar copies that learn from labeled examples and provide a contrast signal. |

The success and failure teachers do **not** replace the actor. Their job is to
provide additional information about *why* a sampled answer looks good or bad.

## 2. Where the training examples come from

For each prompt, the actor samples multiple responses. A verifier then gives
each response one binary semantic outcome:

```text
1 = success (for example, a correct answer)
0 = failure (for example, an incorrect answer)
```

The system stores the prompt and response in one of two replay buffers:

```text
success buffer: prompt -> correct response
failure buffer: prompt -> incorrect response
```

The label is the verifier's semantic result. It is not inferred from a shaped
reward or from individual token rewards.

A prompt can appear in both buffers. For example:

```text
Problem P:
  response 1 -> success
  response 2 -> failure
  response 3 -> failure
```

This is especially useful because the two teachers can compare different
answers to the same problem.

## 3. How the two teachers are trained

### Initialization

At SDC initialization, both teachers start as independent copies of the actor:

```text
theta_0 = actor parameters at initialization
pi_S <- theta_0
pi_F <- theta_0
```

They start equal. They become different only because they see different
examples.

### Periodic supervised updates

Every few actor policy steps, the trainer refreshes both teachers:

1. Take recent examples from the success and failure buffers.
2. Prefer success/failure examples from the same prompt when pairing records.
3. Keep the number of examples and response-token budget balanced.
4. Run response-only supervised fine-tuning on the success teacher.
5. Run response-only supervised fine-tuning on the failure teacher.
6. Mix each teacher slightly back toward the original snapshot `theta_0`.

The supervised objective for either teacher is ordinary language-model
cross-entropy on the response tokens:

```text
L_SFT = average over examples of
        - (sum of response-token log probabilities)
          / (number of response tokens)
```

Prompt tokens are context. They are not counted as prediction targets.

The success teacher therefore becomes better at assigning probability to
patterns found in successful answers. The failure teacher becomes better at
assigning probability to patterns found in failed answers.

### Why mix back toward the original model?

After each refresh, the teachers are softly pulled toward the common starting
model:

```text
teacher <- 0.9 * teacher + 0.1 * theta_0     # default gamma = 0.9
```

This prevents either teacher from drifting too far or becoming a poor language
model. It keeps the comparison focused on success versus failure rather than
on unrelated changes.

### Important distinction

The SFT loss updates the **teachers**. It does not directly update the actor.
The actor learns from the teachers' probability comparison inside the actor
loss described below.

## 4. What the teachers measure

For a token `t` in a failed response, the trainer evaluates that same token
under both teachers:

```text
c_t = log pi_F(token_t) - log pi_S(token_t)
```

Interpretation:

- `c_t > 0`: the failure teacher likes the token more; it looks failure-like.
- `c_t < 0`: the success teacher likes the token more; it looks success-like.
- `c_t ~= 0`: the teachers do not distinguish the token clearly.

The contrast is centered and divided by its RMS scale so that one unusually
large log-probability difference cannot dominate the whole update.

Teacher log probabilities are detached. In other words, gradients do not flow
back into the teachers during the actor update.

SDC scores failed rows because those rows are the ones that need corrective
information. Successful rows continue to use the ordinary RL objective.

## 5. The ordinary PPO/GRPO signal

The actor also receives the normal policy-gradient signal. PPO compares the
current actor with the old policy using an importance ratio:

```text
r_t = pi_theta(token_t) / pi_old(token_t)
```

The advantage `A_t` says which direction to move:

- `A_t > 0`: make this sampled behavior more likely;
- `A_t < 0`: make this sampled behavior less likely;
- `A_t = 0`: do not move from the ordinary policy loss.

GRPO/Dr.GRPO usually centers rewards within each prompt's group. This gives a
relative signal: a response is good or bad compared with the other sampled
responses for that prompt.

PPO then clips the ratio (and this repository also uses dual clipping) so one
batch cannot change the policy too aggressively.

## 6. The SDC-TR loss used by the current run

The `sdc_tr` mode folds the teacher contrast into the PPO ratio rather than
adding a completely separate gradient:

```text
r_sdc,t = exp(
    log pi_theta(token_t) - log pi_old(token_t)
    + lambda_eff * c_t
)
```

where:

```text
lambda_eff = (1 - sdc_tr_alpha) * reliability_gate
```

This modified ratio is then passed through the normal PPO clipping and
dual-clipping path:

```text
L_actor = PPO_loss(A_t, r_sdc,t)
```

The teacher term is active only on failed-response tokens. Everywhere else,
the loss is the ordinary PPO loss.

### What does `sdc_tr_alpha` do?

It controls how much of the teacher contrast is used:

- `alpha = 1.0`: no SDC-TR tilt; this is ordinary PPO exactly.
- lower `alpha`: stronger success/failure tilt.
- `reliability_gate = 0`: the tilt is disabled because the teachers are not
  currently trusted.

The current arm uses `alpha = 0.5`, so the maximum effective tilt before the
reliability gate is applied is `0.5 * c_t`.

### Intuition for the tilt

Failed responses normally have negative advantages. For those tokens:

- if `c_t > 0`, the failure teacher makes the PPO ratio larger, making the
  penalty for retaining that failure-like token stronger;
- if `c_t < 0`, the success teacher wins, so the penalty is reduced and the
  actor is less discouraged from using that token.

The contrast does not bypass PPO. The usual PPO and dual-clip limits still
bound the update.

### Additive SDC versus SDC-TR

The repository also supports an additive `sdc` mode. It keeps the ordinary PPO
loss and adds a separate contrast term:

```text
L_actor = L_PPO + beta * reliability_gate * average(r_t * c_t)
```

The sign has the same meaning: minimizing this term lowers failure-like token
probabilities (`c_t > 0`) and raises success-like ones (`c_t < 0`). The current
Blackwell run uses `sdc_tr`, not additive `sdc`, so its contrast strength is
controlled mainly by `sdc_tr_alpha` and the PPO ratio path.

## 7. Why all-wrong groups need a special repair

Suppose the actor samples eight answers for a difficult problem and all eight
are wrong. Their rewards are identical. After group-centering, every advantage
becomes zero:

```text
A_t = 0 for every response token in the group
```

The ordinary PPO term is proportional to `A_t`, so this group produces no
policy gradient. The SDC-TR tilt alone cannot fix it because it still
multiplies the zero advantage.

That is a serious blind spot: the hardest prompts can be exactly the prompts
that provide no learning signal early in training.

### The zero-advantage repair

When enabled, the repair adds a small, detached pseudo-advantage only when all
of these conditions hold:

- the response is failed;
- the token is a response token;
- the original advantage is exactly zero;
- both teachers are available.

The effective advantage becomes:

```text
mu_eff = sdc_tr_degenerate_coef * reliability_gate
A_eff,t = A_t - mu_eff * c_t       when A_t == 0 on an active failed token
A_eff,t = A_t                      otherwise
```

For the current run:

```text
sdc_tr_degenerate_coef = 0.5
```

The pseudo-advantage goes through the normal PPO clipping and dual clipping.
It is not an unbounded extra loss.

### Sign example

| Teacher comparison | `c_t` | Pseudo-advantage | Actor tendency |
|---|---:|---:|---|
| Failure teacher likes token more | positive | negative | lower the token probability |
| Success teacher likes token more | negative | positive | raise the token probability |

Thus an all-wrong group can still teach the actor which local behavior looks
failure-like and which behavior resembles successful answers.

The repair does not affect groups that already have a nonzero advantage. With
`sdc_tr_degenerate_coef = 0.0` (the default), the repair is disabled and the
previous SDC-TR behavior is preserved.

## 8. How one training cycle fits together

```text
1. Actor generates several responses per prompt.
2. Verifier labels each response success or failure.
3. GRPO/Dr.GRPO computes group-relative advantages.
4. Labeled responses enter the two teacher buffers.
5. Failed responses are scored by both teachers.
6. The trainer computes c_t = log pi_F - log pi_S.
7. SDC-TR modifies the PPO ratio on failed tokens.
8. Zero-advantage repair supplies A_eff where A == 0, if enabled.
9. PPO clipping/dual clipping limits the update.
10. Actor parameters are updated.
11. Periodically, both teachers receive balanced SFT refreshes.
12. The refreshed teachers are used in later actor updates.
```

This creates a feedback loop:

```text
better actor -> more informative successes/failures -> better teachers
            -> better contrast signal -> better actor
```

## 9. Why this can improve learning

A reward tells the actor **which whole response won**. The teacher contrast
adds token-level information about **which parts of a failed response resemble
failure behavior** and which parts resemble successful behavior.

This helps in three ways:

1. **More informative failed-response updates.** The actor can learn from a
   failed sample instead of treating the entire response as an undifferentiated
   mistake.
2. **A signal for all-wrong groups.** The zero-advantage repair prevents hard
   groups from disappearing completely from the update.
3. **Safer updates.** The contrast is filtered by a reliability gate and passed
   through PPO's existing trust-region controls.

SDC is not a proof checker and does not guarantee that every failure is fixed.
It supplies a direction that complements the verifier reward. The verifier
still determines success and failure, and PPO/GRPO remains the main learning
objective.

## 10. What to monitor in W&B

Useful health metrics include:

- `sdc/success_model_update_count` and
  `sdc/failure_model_update_count`: both should advance together;
- `sdc/quality_auc_proxy`: whether the teachers separate failed from successful
  behavior (around `0.5` means no useful separation);
- `sdc/reliability_gate`: how much the trainer trusts the contrast;
- `sdc_tr/active_failed_token_fraction`: how much of the batch receives SDC-TR;
- `sdc_tr/degenerate_token_fraction`: how much zero-advantage repair is used;
- `sdc_tr/pseudo_advantage_abs_mean`: strength of the repair signal;
- PPO KL and clip fractions: whether the actor update remains controlled.

A falling reliability gate is a warning that the teachers are not currently
providing a trustworthy distinction. In that case, SDC-TR automatically fades
toward ordinary PPO.

## One-sentence summary

The actor learns from reward, while two periodically refreshed sidecar teachers
learn what success and failure look like; SDC-TR uses their detached,
normalized probability difference to steer failed responses through the normal
PPO trust region, including a small repair signal for all-wrong groups.
