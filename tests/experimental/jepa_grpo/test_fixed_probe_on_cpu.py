# Why _fixed_probe exists: a per-BATCH probe series cannot resolve a representation
# change, because redrawing problems moves the score on its own. Measured on the real
# no-JEPA baseline, held-out AUROC swung 0.53-0.82 across 130 steps with no trend.
# Here the encoder is held CONSTANT and only the batch is redrawn, so every point of
# spread below is pure confound -- and freezing the rows removes it.
import numpy as np
import sys

sys.path.insert(0, "/workspace/Joint-Embedding-Guided-Policy-Optimization")
from verl.experimental.jepa_grpo.ray_trainer import _separability, _within_prompt_cos

D, N_PROMPTS, K = 64, 24, 8          # dims, prompts per batch, rollouts per prompt


def _draw_batch(rng, encoder, difficulty):
    """One step's rollouts. `difficulty` shifts p(correct) -- that is the batch effect."""
    uids, correct, raw = [], [], []
    for p in range(N_PROMPTS):
        p_correct = np.clip(rng.normal(difficulty, 0.15), 0.05, 0.95)
        for _ in range(K):
            ok = rng.random() < p_correct
            v = rng.standard_normal(D)
            v[0] += 1.6 if ok else -1.6      # the SAME correctness signal every time
            uids.append(str(p)); correct.append(ok); raw.append(v)
    return (np.asarray(raw) @ encoder, np.asarray(correct), np.asarray(uids, dtype=object))


def _auroc(X, correct, uids):
    return _separability(X, correct, uids)[1]


def test_fixed_rows_remove_the_batch_confound():
    rng = np.random.default_rng(0)
    encoder = rng.standard_normal((D, D))    # constant: representation never changes

    # Protocol A (what the per-batch figure does): new problems each step.
    per_batch = [_auroc(*_draw_batch(rng, encoder, d)) for d in (0.3, 0.5, 0.7, 0.4, 0.6)]

    # Protocol B (_fixed_probe): snapshot once, re-encode the same rows every step.
    X, correct, uids = _draw_batch(rng, encoder, 0.5)
    fixed = [_auroc(X, correct, uids) for _ in range(5)]

    spread_batch = max(per_batch) - min(per_batch)
    spread_fixed = max(fixed) - min(fixed)
    assert spread_fixed < 1e-9, fixed                     # identical encoder -> identical score
    assert spread_batch > 0.05, per_batch                 # redrawing alone moves it
    assert spread_batch > 10 * max(spread_fixed, 1e-12)


def test_fixed_probe_still_tracks_a_real_representation_change():
    # Freezing rows must not freeze the metric: a genuinely worse encoder must score worse.
    rng = np.random.default_rng(1)
    good = rng.standard_normal((D, D))
    X, correct, uids = _draw_batch(rng, good, 0.5)

    bad = good.copy()
    bad[0] = 0.0                                          # delete the informative direction
    raw = X @ np.linalg.pinv(good)                        # recover pre-encoder rows
    assert _auroc(raw @ bad, correct, uids) < _auroc(X, correct, uids) - 0.1


def test_within_prompt_cos_excludes_cross_prompt_pairs():
    # Two prompts, orthogonal to each other; within each, correct==correct and wrong is
    # opposite. Cross-prompt pairs (cosine 0) must not leak in.
    p = np.array([[1., 0., 0., 0.], [1., 0., 0., 0.], [-1., 0., 0., 0.],
                  [0., 1., 0., 0.], [0., 1., 0., 0.], [0., -1., 0., 0.]])
    correct = np.array([True, True, False, True, True, False])
    uids = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)
    cc, cw = _within_prompt_cos(p, correct, uids)
    assert abs(cc - 1.0) < 1e-9, cc
    assert abs(cw + 1.0) < 1e-9, cw       # -1, not the 0 a cross-prompt average would give


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"pass {name}")
    print("ok")
