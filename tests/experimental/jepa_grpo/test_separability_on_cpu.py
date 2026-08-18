# _separability must report HELD-OUT AUROC: ~0.5 on noise (where the in-sample
# probe scores a perfect 1.0), high on a genuinely separable direction.
import numpy as np, sys
sys.path.insert(0, "/workspace/Joint-Embedding-Guided-Policy-Optimization")
from verl.experimental.jepa_grpo.ray_trainer import _separability

rng = np.random.default_rng(0)
n, d = 256, 1536
groups = np.repeat(np.arange(32), 8).astype(str)
y = rng.random(n) < 0.45

noise = rng.standard_normal((n, d))
_, auc_noise = _separability(noise, y, groups)

signal = noise.copy()
signal[y, 0] += 6.0                       # one genuinely informative dimension
_, auc_signal = _separability(signal, y, groups)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ins = LogisticRegression(max_iter=2000).fit(noise, y.astype(int))
auc_in = roc_auc_score(y.astype(int), ins.decision_function(noise))

assert auc_in > 0.99, auc_in                      # in-sample shatters pure noise
assert 0.35 < auc_noise < 0.65, auc_noise         # held-out correctly says "nothing here"
assert auc_signal > 0.65, auc_signal              # and still finds real structure
assert auc_signal - auc_noise > 0.15, (auc_signal, auc_noise)
print(f"in-sample on noise={auc_in:.3f}  held-out on noise={auc_noise:.3f}  held-out on signal={auc_signal:.3f}")
print("ok")
