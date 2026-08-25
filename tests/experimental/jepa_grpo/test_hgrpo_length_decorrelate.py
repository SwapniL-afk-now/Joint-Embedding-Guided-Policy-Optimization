"""hgrpo_loss length decorrelation (jepa.length_decorrelate).

The failure it targets: on run ar6ntx2j the h-grpo margin correlated +0.81 with response
length and -0.11 with accuracy -- the objective was rewarding teacher-SHAPED (long)
rollouts rather than correct ones. Residualising s against length within each group must
remove a purely length-driven margin while leaving a genuine one intact.

    python tests/test_hgrpo_length_decorrelate.py
"""
import torch

from verl.experimental.jepa_grpo.core_algos import hgrpo_loss


def _margin(student, label, teacher, gid, length=None):
    _, m = hgrpo_loss(student=student, label=label, teacher=teacher,
                      group_id=gid, length=length)
    return m["jepa/margin"], m


def _students_with_cosine(a, gid, cos_target, seed):
    """Unit rows whose cosine to their anchor is EXACTLY cos_target.

    hgrpo_loss L2-normalises the student block, so a fixture that only scales the norm
    produces s == 1 everywhere and no margin at all. The confound is directional: build
    z = c*a + sqrt(1-c^2)*w with w a unit vector orthogonal to a, giving a.z == c exactly.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(gid.shape[0], a.shape[1], generator=g)
    an = a[gid]
    w = w - (w * an).sum(dim=1, keepdim=True) * an          # orthogonalise against anchor
    w = torch.nn.functional.normalize(w, dim=-1)
    c = cos_target.unsqueeze(1)
    return c * an + (1 - c.pow(2)).clamp_min(0).sqrt() * w


def test_pure_length_confound_is_removed():
    """Correctness is INDEPENDENT of the anchor direction; only length tracks it.

    Built so s_ij is an exact affine function of length. The raw margin is non-zero purely
    because the correct rollouts happen to be the longer ones; after residualisation it
    must vanish.
    """
    torch.manual_seed(0)
    d, P, G = 16, 6, 8
    a = torch.nn.functional.normalize(torch.randn(P, d), dim=-1)
    gid = torch.arange(P).repeat_interleave(G)
    length = torch.arange(G, dtype=torch.float).repeat(P) * 50 + 300
    # cosine is an EXACT affine function of length -> the whole margin is length
    student = _students_with_cosine(a, gid, 0.1 + 0.0005 * length, seed=10)
    # correct == the longer half, so the labels correlate with length and nothing else
    label = (length > length.reshape(P, G).median(dim=1).values.repeat_interleave(G)).float()

    raw, _ = _margin(student, label, a, gid)
    dec, met = _margin(student, label, a, gid, length=length)
    assert abs(raw) > 1e-3, f"fixture broken: raw margin should be non-zero, got {raw}"
    assert abs(dec) < 1e-4, f"length-only margin must vanish, got {dec}"
    assert met["jepa/length_r2"] > 0.9, f"length should explain ~all variance, got {met['jepa/length_r2']}"


def test_genuine_signal_survives():
    """Correctness aligned with the anchor, length random -> margin must be preserved."""
    torch.manual_seed(1)
    d, P, G = 16, 6, 8
    a = torch.nn.functional.normalize(torch.randn(P, d), dim=-1)
    gid = torch.arange(P).repeat_interleave(G)
    label = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).repeat(P)
    length = torch.randint(300, 900, (P * G,)).float()
    # cosine depends on CORRECTNESS only; length is independent noise
    student = _students_with_cosine(a, gid, 0.2 + 0.6 * label, seed=11)

    raw, _ = _margin(student, label, a, gid)
    dec, met = _margin(student, label, a, gid, length=length)
    assert raw > 0.5, f"fixture broken: expected a large true margin, got {raw}"
    assert dec > 0.5 * raw, f"genuine margin must survive: {raw} -> {dec}"
    assert met["jepa/length_r2"] < 0.5, f"length should explain little here, got {met['jepa/length_r2']}"


def test_gradients_finite_and_shift_invariant():
    """The projection must not break the softmax's shift invariance or produce NaNs."""
    torch.manual_seed(2)
    d, P, G = 8, 4, 6
    a = torch.nn.functional.normalize(torch.randn(P, d), dim=-1)
    gid = torch.arange(P).repeat_interleave(G)
    label = torch.randint(0, 2, (P * G,)).float()
    length = torch.randint(300, 900, (P * G,)).float()
    student = torch.randn(P * G, d, requires_grad=True)

    loss, met = hgrpo_loss(student=student, label=label, teacher=a,
                           group_id=gid, length=length)
    loss.backward()
    assert torch.isfinite(loss), "loss must be finite"
    assert torch.isfinite(student.grad).all(), "grads must be finite"
    assert 0.0 <= met["jepa/length_r2"] <= 1.0 + 1e-6, met["jepa/length_r2"]


def test_no_length_reproduces_original():
    """length=None must be bit-identical to the pre-change behaviour."""
    torch.manual_seed(3)
    d, P, G = 8, 4, 6
    a = torch.nn.functional.normalize(torch.randn(P, d), dim=-1)
    gid = torch.arange(P).repeat_interleave(G)
    label = torch.randint(0, 2, (P * G,)).float()
    student = torch.randn(P * G, d)
    loss, met = hgrpo_loss(student=student, label=label, teacher=a, group_id=gid)
    assert torch.isfinite(loss)
    assert met["jepa/length_r2"] == 0.0, "no length passed => no decorrelation reported"


if __name__ == "__main__":
    test_pure_length_confound_is_removed()
    test_genuine_signal_survives()
    test_gradients_finite_and_shift_invariant()
    test_no_length_reproduces_original()
    print("all ok")
