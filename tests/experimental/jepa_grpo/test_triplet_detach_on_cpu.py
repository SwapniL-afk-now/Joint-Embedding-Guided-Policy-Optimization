# llm-jepa-triplet stop-grad checks. The claim that must hold:
#
#   With the teacher anchor LIVE, the objective has a shortcut -- move the anchor into the
#   student cloud. That raises cos(p+,a) and cos(p-,a) equally, reduces `align`, and leaves
#   the margin at zero. That is not a hypothesis: run 0ntqn3zc did exactly this for 170
#   steps (cos_correct 0.8525->0.9393, cos_wrong 0.8550->0.9445, violation pinned at 1.00).
#
#   detach_teacher=True must (a) send no gradient to the anchor at all, and (b) make the
#   uniform-contraction shortcut stop paying -- a step that moves both students equally
#   closer to the anchor must NOT reduce the repel term.
import sys

import torch

sys.path.insert(0, "/workspace/Joint-Embedding-Guided-Policy-Optimization")
from verl.experimental.jepa_grpo.core_algos import llm_jepa_triplet_loss


def _call(correct, wrong, teacher, detach, margin=0.0, repel_weight=1.0):
    return llm_jepa_triplet_loss(
        correct=correct, wrong=wrong, teacher=teacher,
        group_id=torch.arange(correct.shape[0]),
        tau=0.1, margin=margin, repel_weight=repel_weight,
        include_cot=True, detach_teacher=detach,
    )


def test_detach_blocks_gradient_to_the_anchor_only():
    torch.manual_seed(0)
    d, m = 16, 6
    for detach, anchor_gets_grad in ((True, False), (False, True)):
        c = torch.randn(m, d, requires_grad=True)
        w = torch.randn(m, d, requires_grad=True)
        a = torch.randn(m, d, requires_grad=True)
        loss, _ = _call(c, w, a, detach)
        loss.backward()
        got = a.grad is not None and a.grad.abs().sum() > 0
        assert got == anchor_gets_grad, (detach, got)
        # The students must keep learning either way -- detaching the anchor must not
        # accidentally cut the whole term.
        assert c.grad.abs().sum() > 0 and w.grad.abs().sum() > 0, detach


def test_contraction_plateaus_at_margin_zero_and_cannot_cross_it():
    # THE failure mode, and it is subtler than "contraction does not pay". Sliding BOTH
    # students toward the anchor (the lockstep rise measured in 0ntqn3zc) DOES reduce the
    # hinge -- it compresses every cosine difference toward zero, so a negative margin
    # shrinks in magnitude. What it can never do is make the margin POSITIVE. The hinge
    # therefore asymptotes at tau*softplus(0) = tau*log2 and stops, a plateau that only
    # real separation escapes.
    #
    # This is where run 0ntqn3zc sat: with margin~0 and margin param 0.1, the hinge floor
    # is tau*softplus(1) = 0.1313, and the run reported jepa/repel_teacher = 0.1338 for
    # 170 steps. It was parked on the plateau, not descending toward it.
    torch.manual_seed(0)
    d, m = 32, 8
    a = torch.nn.functional.normalize(torch.randn(m, d), dim=-1)
    c0 = torch.nn.functional.normalize(torch.randn(m, d), dim=-1)
    w0 = torch.nn.functional.normalize(torch.randn(m, d), dim=-1)

    def contract(x, t):          # slide x toward a by fraction t
        return torch.nn.functional.normalize((1 - t) * x + t * a, dim=-1)

    def probe(t):
        _, mt = _call(contract(c0, t), contract(w0, t), a.clone(), True, repel_weight=1.0)
        return mt["jepa/repel_teacher"], mt["jepa/margin"]

    floor = 0.1 * torch.log(torch.tensor(2.0)).item()      # tau * softplus(0)
    seq = [probe(t) for t in (0.0, 0.5, 0.95, 0.999)]
    margins = [mg for _, mg in seq]
    repels = [r for r, _ in seq]

    # Contraction never flips the sign: the margin approaches 0 from below and stops.
    assert all(mg <= 1e-4 for mg in margins), margins
    assert abs(margins[-1]) < 1e-3, margins            # pinned at zero, not positive
    # And the hinge bottoms out at the margin==0 value; it cannot get under it this way.
    assert repels[-1] >= floor - 1e-4, (repels, floor)
    assert repels[0] > repels[-1], repels              # partial payoff is real

    # Only genuine separation gets below the plateau.
    _, sep = _call(a.clone(), -a.clone(), a.clone(), True, repel_weight=1.0)
    assert sep["jepa/margin"] > 1.5, sep["jepa/margin"]
    assert sep["jepa/repel_teacher"] < floor / 2, (sep["jepa/repel_teacher"], floor)


def test_margin_zero_makes_violation_the_retrieval_error():
    # At margin=0, 1 - violation IS nearest-anchor retrieval accuracy over {p+, p-}.
    # This is the metric requested; it needs no new code, only margin=0.
    a = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    good, bad = a.clone(), -a.clone()
    _, mt = _call(good, bad, a.clone(), True, margin=0.0)
    assert mt["jepa/violation"] == 0.0, mt          # all 4 retrieved correctly
    _, mt = _call(bad, good, a.clone(), True, margin=0.0)
    assert mt["jepa/violation"] == 1.0, mt          # all 4 retrieved wrongly


def test_empty_and_default_paths_unchanged():
    # detach_teacher defaults False, so every existing arm keeps its exact behaviour.
    z = torch.zeros(0, 8)
    loss, mt = llm_jepa_triplet_loss(
        correct=z, wrong=z, teacher=z, group_id=torch.zeros(0, dtype=torch.long))
    assert float(loss) == 0.0 and mt["jepa/margin"] == 0.0

    torch.manual_seed(0)
    c, w, a = (torch.randn(5, 12, requires_grad=True) for _ in range(3))
    l_default, _ = llm_jepa_triplet_loss(
        correct=c, wrong=w, teacher=a, group_id=torch.arange(5), tau=0.1, margin=0.1)
    l_explicit, _ = _call(c, w, a, False, margin=0.1)
    assert torch.allclose(l_default, l_explicit)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"pass {name}")
    print("ok")
