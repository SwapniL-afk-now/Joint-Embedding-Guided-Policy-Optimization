# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU checks for core_algos.hgrpo_loss (loss_type='h-grpo').

Each test kills one specific way the previous representation arms failed.
"""

import types

import numpy as np
import torch
import torch.nn.functional as F

from verl.experimental.jepa_grpo.core_algos import hgrpo_loss, llm_jepa_triplet_loss
from verl.experimental.jepa_grpo.ray_trainer import JEPARayPPOTrainer


def _group(n_correct: int, n_wrong: int, d: int = 16, seed: int = 0):
    """One prompt: an anchor plus n_correct/n_wrong student reads around it."""
    g = torch.Generator().manual_seed(seed)
    M = n_correct + n_wrong
    teacher = torch.randn(1, d, generator=g)
    student = torch.randn(M, d, generator=g, requires_grad=True)
    label = torch.tensor([1.0] * n_correct + [0.0] * n_wrong)
    group_id = torch.zeros(M, dtype=torch.long)
    return student, label, teacher, group_id


def _scores(student, teacher, group_id):
    a = F.normalize(teacher, dim=-1)
    return (F.normalize(student, dim=-1) * a[group_id]).sum(-1)


def test_common_mode_shift_is_invisible_to_hgrpo_but_not_to_the_triplet():
    """The failure mode of run 0ntqn3zc: raising every cosine by the same amount.

    H-GRPO's softmax is exactly invariant to s -> s + gamma, so both the loss AND the
    gradient are unchanged. llm_jepa_triplet_loss, which carries an absolute
    `1 - cos(p+, a)` align term, is measurably rewarded for the same move -- which is
    why it could descend for 170 steps with the margin pinned at zero.
    """
    student, label, teacher, gid = _group(3, 3, seed=1)
    a = F.normalize(teacher, dim=-1)[0]

    def loss_and_grad(x):
        x = x.detach().clone().requires_grad_(True)
        loss, _ = hgrpo_loss(x, label, teacher, gid)
        loss.backward()
        return float(loss), x.grad.clone()

    base_loss, base_grad = loss_and_grad(student)

    # Move every student the SAME way along the anchor: pure common mode, no change to
    # any pairwise difference. Renormalise so only the cosines moved.
    shifted = F.normalize(F.normalize(student.detach(), dim=-1) + 0.3 * a, dim=-1)
    s0 = _scores(student.detach(), teacher, gid)
    s1 = _scores(shifted, teacher, gid)
    assert (s1 > s0).all(), "the perturbation must actually raise every cosine"

    shift_loss, shift_grad = loss_and_grad(shifted)
    assert abs(shift_loss - base_loss) < 1e-5, (
        f"H-GRPO must be blind to the common mode; {base_loss} -> {shift_loss}"
    )
    assert base_grad.abs().sum() > 1e-6, "guard: the unshifted gradient must be non-trivial"

    # The triplet loss, same perturbation, is strictly rewarded for it.
    tri_kw = dict(group_id=gid[:3], tau=0.1, margin=0.0, detach_teacher=True)
    tri_base, _ = llm_jepa_triplet_loss(
        correct=student.detach()[:3], wrong=student.detach()[3:],
        teacher=teacher.expand(3, -1), **tri_kw)
    tri_shift, _ = llm_jepa_triplet_loss(
        correct=shifted[:3], wrong=shifted[3:],
        teacher=teacher.expand(3, -1), **tri_kw)
    assert float(tri_shift) < float(tri_base) - 1e-3, (
        "the triplet loss should reward the common mode -- that is the bug H-GRPO fixes"
    )


def test_gradient_step_strictly_increases_the_correct_wrong_gap():
    """Spec section 9: d/dt (s+ - s-) > 0 for a mixed group."""
    student, label, teacher, gid = _group(4, 4, seed=2)
    x = student.detach().clone().requires_grad_(True)
    gaps = []
    for _ in range(5):
        s = _scores(x, teacher, gid)
        gaps.append(float(s[label > 0.5].mean() - s[label < 0.5].mean()))
        loss, _ = hgrpo_loss(x, label, teacher, gid)
        (grad,) = torch.autograd.grad(loss, x)
        x = (x - 10.0 * grad).detach().requires_grad_(True)
    s = _scores(x, teacher, gid)
    gaps.append(float(s[label > 0.5].mean() - s[label < 0.5].mean()))
    assert all(b > a for a, b in zip(gaps, gaps[1:])), f"gap must rise monotonically: {gaps}"


def test_all_wrong_group_is_inert_and_must_not_be_rescued():
    """An all-wrong group carries no discriminative signal and must contribute NOTHING.

    Rescuing it with a constant teacher column (s=1, no gradient) makes the only
    available descent direction "push every student score down together" — the exact
    common mode this objective exists to exclude, and unbounded, since every student
    carries A < 0 and nothing pulls back. It leaks through the shared encoder to every
    rollout: measured -0.12 mean score drift per step here vs -0.0025 for a mixed group.
    Cost two runs (io1aoneh, d512pwma) before it was removed.
    """
    student, _, teacher, gid = _group(0, 6, seed=3)
    label = torch.zeros(6)
    x = student.detach().clone().requires_grad_(True)
    loss, _ = hgrpo_loss(x, label, teacher, gid)
    (grad,) = torch.autograd.grad(loss, x)
    assert grad.abs().sum() == 0.0, "an all-wrong group must produce no gradient"

    # ...and specifically no common-mode drift: scores must not move at all.
    s_before = _scores(x.detach(), teacher, gid)
    s_after = _scores(x.detach() - 10.0 * grad, teacher, gid)
    assert torch.allclose(s_before, s_after), f"{s_before} -> {s_after}"

    # The reason it is inert: uniform labels => sigma = 0 => every advantage vanishes.
    q = torch.softmax(_scores(x.detach(), teacher, gid), dim=0)
    cbar = (q * label).sum()
    assert float(((label - cbar) ** 2 * q).sum()) == 0.0


def test_anchor_is_stop_grad_and_students_are_not():
    student, label, teacher, gid = _group(2, 2, seed=4)
    t = teacher.detach().clone().requires_grad_(True)
    x = student.detach().clone().requires_grad_(True)
    loss, _ = hgrpo_loss(x, label, t, gid)
    loss.backward()
    assert t.grad is None, "a_i = sg(f_theta(y^T)); the anchor must receive no gradient"
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_multi_prompt_ragged_groups_and_all_correct_degeneracy():
    """Ragged group sizes must scatter correctly; an all-correct group contributes 0."""
    d = 16
    g = torch.Generator().manual_seed(5)
    # prompt 0: 3 rollouts (2 correct), prompt 1: 5 rollouts (0 correct -> inert),
    # prompt 2: 2 rollouts (2 correct -> inert). The builder drops 1 and 2; the loss
    # must stay finite and give them zero gradient if they reach it anyway.
    sizes, labels = [3, 5, 2], [[1.0, 1.0, 0.0], [0.0] * 5, [1.0, 1.0]]
    student = torch.randn(sum(sizes), d, generator=g, requires_grad=True)
    teacher = torch.randn(3, d, generator=g)
    gid = torch.tensor([0] * 3 + [1] * 5 + [2] * 2)
    label = torch.tensor([v for row in labels for v in row])

    loss, m = hgrpo_loss(student, label, teacher, gid)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert m["jepa/n_allwrong_groups"] == 1
    # Only prompt 0 is mixed, so the margin/violation stats come from it alone.
    s = _scores(student.detach(), teacher, gid)
    assert abs(m["jepa/margin"] - float(s[:2].mean() - s[2])) < 1e-5
    # The all-correct group (prompt 2) has zero advantage -> zero gradient on its rows.
    assert student.grad[-2:].abs().sum() < 1e-7


# --------------------------------------------------- builder -> loss contract


def _fake_trainer(teacher_texts):
    return types.SimpleNamespace(
        teacher_targets=teacher_texts,
        tokenizer=types.SimpleNamespace(pad_token_id=0, eos_token_id=None),
        jepa_cfg=types.SimpleNamespace(min_valid_pairs=1),
        _tokenize_target=lambda problem, text, view: torch.tensor([7, 8, 9]),
        _template_prefix_ids=lambda: [],
    )


def _fake_batch(rewards_per_prompt, prompt_len=3):
    """rewards_per_prompt: list of per-prompt reward lists, one entry per rollout."""
    flat = [r for row in rewards_per_prompt for r in row]
    rows = len(flat)
    ids = torch.arange(1, rows * prompt_len + 1).reshape(rows, prompt_len)
    uids, extra = [], []
    for p, row in enumerate(rewards_per_prompt):
        for _ in row:
            uids.append(f"p{p}")
            extra.append({"index": p, "problem": f"q{p}"})
    batch = types.SimpleNamespace(
        batch={"input_ids": ids, "attention_mask": torch.ones(rows, prompt_len, dtype=torch.long),
               "responses": ids},
        non_tensor_batch={"uid": np.array(uids, dtype=object),
                          "extra_info": np.array(extra, dtype=object)},
    )
    reward = torch.tensor(flat, dtype=torch.float32).reshape(-1, 1)
    return batch, reward, np.array(["cot"] * rows, dtype=object)


def test_builder_drops_every_non_mixed_group():
    """Only mixed groups carry signal. All-correct AND all-wrong are dropped before the
    forward — the all-wrong ones because rescuing them costs a common-mode collapse
    (see test_all_wrong_group_is_inert_and_must_not_be_rescued), and dropping them here
    saves the forward pass entirely."""
    # p0 mixed (2/2), p1 all-wrong (dropped), p2 all-correct (dropped), p3 mixed (1/3)
    batch, reward, view = _fake_batch([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    fake = _fake_trainer({0: ["t0"], 1: ["t1"], 2: ["t2"], 3: ["t3"]})
    out = JEPARayPPOTrainer._build_jepa_batch_hgrpo(fake, batch, reward, view)

    assert out is not None
    assert out.meta_info["n_groups"] == 2, "only the two mixed prompts survive"
    assert out.meta_info["n_allwrong_groups"] == 1, "counts the all-wrong prompt DROPPED"
    assert out.meta_info["n_anchors"] == 8        # 2 surviving prompts x 4 rollouts
    assert out.meta_info["n_unique_targets"] == 2  # one anchor per surviving prompt
    assert out.batch["student_label"][:8].tolist() == [1, 1, 0, 0] + [1, 0, 0, 0]
    assert out.batch["student_group_id"][:8].tolist() == [0] * 4 + [1] * 4


def test_worker_unpacking_of_the_builder_output_feeds_the_loss():
    """Replays worker.jepa_update's h-grpo branch against a stubbed encoder."""
    batch, reward, view = _fake_batch([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    fake = _fake_trainer({0: ["t0"], 1: ["t1"], 2: ["t2"], 3: ["t3"]})
    d = JEPARayPPOTrainer._build_jepa_batch_hgrpo(fake, batch, reward, view).to_tensordict()

    M = int((d["anchor_lengths"].long() > 0).sum())
    U = int((d["target_lengths"].long() > 0).sum())
    tgi = d["tgt_cot_idx"].long()
    tgi = tgi[tgi >= 0]
    label = d["student_label"].float()[:M]
    gid = d["student_group_id"].long()[:M]
    assert int(gid.max()) + 1 == tgi.shape[0], "one teacher anchor per group"

    torch.manual_seed(0)
    joint = torch.randn(M + U, 16, requires_grad=True)   # stands in for the encoder output
    loss, m = hgrpo_loss(student=joint[:M], label=label,
                         teacher=joint[M:][tgi], group_id=gid)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(joint.grad).all()
    assert joint.grad[:M].abs().sum() > 0, "students must receive gradient"
    assert joint.grad[M:].abs().sum() == 0, "anchors are stop-grad"
    assert m["jepa/n_allwrong_groups"] == 0, "the builder already dropped the all-wrong group"
    assert 0.0 <= m["jepa/violation"] <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")


def test_no_common_mode_drift_in_any_group():
    """Regression for runs io1aoneh and d512pwma. A constant s=1 teacher column carries
    no gradient, so the only way to grow its softmax mass is to push every student score
    down together — the common mode this objective exists to exclude. With the column in
    every group that collapsed cos_cot/cos_neg 0.56->-0.02 and cost 7pp of val by step
    120; restricted to all-wrong groups it did the same thing a third the speed
    (0.57->0.22 by step 124). The column is gone: EVERY group is a students-only softmax,
    which is exactly invariant to s -> s + gamma.

    The operative check is on the SCORES, not the embeddings: L2 normalisation means a
    uniform embedding shift is not a uniform score shift, so an embedding-space assertion
    would test the wrong invariance.
    """
    torch.manual_seed(3)
    d = 8
    teacher = torch.nn.functional.normalize(torch.randn(1, d), dim=-1)
    a = torch.nn.functional.normalize(teacher, dim=-1)[0]

    def mean_score_drift(labels):
        st = torch.nn.functional.normalize(torch.randn(len(labels), d), dim=-1).requires_grad_(True)
        lab = torch.tensor(labels)
        loss, _ = hgrpo_loss(st, lab, teacher, torch.zeros(len(labels), dtype=torch.long))
        loss.backward()
        before = (torch.nn.functional.normalize(st.detach(), dim=-1) * a).sum(-1)
        after = (torch.nn.functional.normalize(st.detach() - st.grad, dim=-1) * a).sum(-1)
        return float((after - before).mean()), st.grad, before, after

    # mixed group: trains, but with no net drift — correct up, wrong down, zero-sum
    drift, grad, before, after = mean_score_drift([1.0, 0.0, 1.0, 0.0])
    assert abs(drift) < 0.02, f"mixed group drifted {drift:+.4f}"
    assert float(grad.abs().sum()) > 0.0, "a mixed group must still train"
    assert (after[[0, 2]] > before[[0, 2]]).all(), "correct rollouts must be pushed up"
    assert (after[[1, 3]] < before[[1, 3]]).all(), "wrong rollouts must be pushed down"

    # all-wrong group: inert. This is where the old teacher column pumped -0.12/step.
    drift, grad, _, _ = mean_score_drift([0.0, 0.0, 0.0])
    assert drift == 0.0 and float(grad.abs().sum()) == 0.0

    # the algebraic reason: sum_j q_j A_j == 0 exactly for a students-only softmax,
    # so dL/ds_j = -q_j A_j has no common component.
    st = torch.nn.functional.normalize(torch.randn(4, d), dim=-1)
    lab = torch.tensor([1.0, 0.0, 1.0, 0.0])
    q = torch.softmax((st * a).sum(-1), dim=0)
    cbar = (q * lab).sum()
    A = (lab - cbar) / ((q * (lab - cbar).pow(2)).sum().sqrt() + 1e-6)
    assert abs(float((q * A).sum())) < 1e-5
